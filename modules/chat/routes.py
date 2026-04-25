"""
Chat routes — Blueprint: streaming AI chat with multi-turn history,
stop generation, edit prompt, session restore, DOCX download.
"""
import io
import json
import logging
import os
import tempfile

from flask import Blueprint, request, jsonify, send_file, send_from_directory, Response

from config import USE_CLOUD, OPENROUTER_MODEL, UPLOAD_FOLDER, rate_limit, _LLM_SEM
from llm.client import _stream_llama_server, _stream_openrouter, get_model, LLM_MODEL
from modules.chat.session import _CHAT_SESSIONS, _CHAT_SYSTEM_PROMPT

logger   = logging.getLogger(__name__)
chat_bp  = Blueprint('chat', __name__)


@chat_bp.route('/chat')
def chat_page():
    return send_from_directory('static', 'chat.html')


@chat_bp.route('/api/chat/stream', methods=['POST'])
@rate_limit("20 per minute; 200 per hour")
def chat_stream():
    data       = request.json or {}
    session_id = data.get('session_id', '').strip()
    message    = data.get('message', '').strip()

    if not message:
        return jsonify({"error": "message required"}), 400
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    try:
        _CHAT_SESSIONS.append_message(session_id, {"role": "user", "content": message})
    except MemoryError as e:
        return jsonify({"error": str(e)}), 503

    history  = _CHAT_SESSIONS.get(session_id)
    messages = [{"role": "system", "content": _CHAT_SYSTEM_PROMPT}] + history
    model    = OPENROUTER_MODEL if USE_CLOUD else (get_model() or LLM_MODEL)

    def generate():
        full_response = []
        try:
            with _LLM_SEM:
                token_gen = (
                    _stream_openrouter(messages)
                    if USE_CLOUD
                    else _stream_llama_server(messages, model)
                )
                for token in token_gen:
                    full_response.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"
            if full_response:
                _CHAT_SESSIONS.append_message(
                    session_id,
                    {"role": "assistant", "content": "".join(full_response)},
                )
            else:
                # LLM yielded zero tokens — remove orphaned user message
                _CHAT_SESSIONS.pop_last_user(session_id)
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("[chat/stream] session=%s error=%s", session_id, e, exc_info=True)
            _CHAT_SESSIONS.pop_last_user(session_id)
            yield f"data: {json.dumps({'error': 'Stream error — please retry'})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        generate(),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@chat_bp.route('/api/chat/session', methods=['GET'])
def get_chat_session():
    """Return stored messages — used by frontend to restore on page reload."""
    session_id = request.args.get('session_id', '').strip()
    messages   = _CHAT_SESSIONS.get(session_id)
    return jsonify({"messages": messages})


@chat_bp.route('/api/chat/truncate', methods=['POST'])
def truncate_chat_session():
    """Keep only the first keep_up_to messages — used by Edit to discard subsequent turns."""
    data       = request.json or {}
    session_id = data.get('session_id', '').strip()
    try:
        keep_up_to = max(0, int(data.get('keep_up_to', 0)))
    except (ValueError, TypeError):
        return jsonify({"error": "keep_up_to must be a non-negative integer"}), 400
    _CHAT_SESSIONS.truncate(session_id, keep_up_to)
    return jsonify({"success": True})


@chat_bp.route('/api/chat/stop', methods=['POST'])
def stop_chat_stream():
    """Client-initiated stop: remove orphaned user message if generation was aborted."""
    data       = request.json or {}
    session_id = data.get('session_id', '').strip()
    _CHAT_SESSIONS.pop_last_user(session_id)
    return jsonify({"success": True})


@chat_bp.route('/api/chat/session', methods=['DELETE'])
def clear_chat_session():
    data       = request.json or {}
    session_id = data.get('session_id', '').strip()
    del _CHAT_SESSIONS[session_id]
    return jsonify({"success": True})


@chat_bp.route('/api/chat/download-docx', methods=['POST'])
def download_chat_docx():
    from docx import Document
    data       = request.json or {}
    session_id = data.get('session_id', '').strip()
    messages   = _CHAT_SESSIONS.get(session_id)

    if not messages:
        return jsonify({"error": "No conversation to download"}), 400

    try:
        doc = Document()
        doc.add_heading("Chat Conversation", 0)
        for msg in messages:
            p = doc.add_paragraph()
            p.add_run(f"{msg['role'].capitalize()}: ").bold = True
            p.add_run(msg['content'])
            doc.add_paragraph()

        # Write to BytesIO — no temp file, nothing to clean up
        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        return send_file(
            buf,
            as_attachment=True,
            download_name='chat_conversation.docx',
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        )
    except Exception as e:
        logger.error("[chat/download-docx] %s", e, exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
