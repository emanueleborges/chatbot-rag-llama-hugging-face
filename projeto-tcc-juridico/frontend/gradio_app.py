"""
Interface Gradio — chat RAG com memória, upload e classificação.

Arranque (raiz projeto-tcc-juridico/):
  pip install -r backend/requirements.txt
  python frontend/gradio_app.py
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parent / ".env", override=True)
load_dotenv(ROOT / ".env", override=True)

import gradio as gr

from backend.chain import chat
from backend.classifier import classificar
from backend.config import GRADIO_PORT, OLLAMA_MODEL
from backend.ingest import extract_bytes, save_upload
from backend.memory import clear_session
from backend.rag_engine import collection_stats, index_all_knowledge, index_file
from backend.training import save_feedback

SESSION = str(uuid.uuid4())[:8]


def respond(message: str, history: list):
    if not (message or "").strip():
        return history, ""
    try:
        out = chat(message.strip(), session_id=SESSION)
        answer = out["answer"]
        if out.get("sources"):
            refs = ", ".join(
                s["source"] for s in out["sources"][:3] if s.get("source")
            )
            if refs:
                answer += f"\n\n*Fontes:* {refs}"
        history = history + [(message, answer)]
    except Exception as e:
        history = history + [(message, f"Erro: {e}")]
    return history, ""


def do_classify(texto: str):
    if len((texto or "").strip()) < 50:
        return "Texto muito curto (mín. 50 caracteres)."
    r = classificar(texto)
    probs = ", ".join(f"{k}: {v}" for k, v in r.get("probabilities", {}).items())
    extra = f" | cobertura tópicos: {r['cobertura_topicos']}" if "cobertura_topicos" in r else ""
    return f"**{r['label']}** ({r['modo']}) — {probs}{extra}"


def do_upload(file):
    if file is None:
        return "Nenhum ficheiro."
    path = Path(file)
    content = path.read_bytes()
    dest = save_upload(path.name, content)
    n = index_file(dest)
    return f"Indexado: {dest.name} → {n} chunks."


def do_index_all():
    stats = index_all_knowledge()
    chroma = collection_stats()
    return f"Indexação: {stats} | Chroma: {chroma}"


def do_feedback(pergunta, resposta, util, correcao):
    if not pergunta.strip():
        return "Indique a pergunta."
    save_feedback(
        pergunta=pergunta,
        resposta=resposta or "",
        util=bool(util),
        correcao=correcao or "",
        session_id=SESSION,
    )
    return "Feedback guardado (refinamento indexado se útil + correcção)."


def do_clear():
    clear_session(SESSION)
    return []


with gr.Blocks(title="Assistente Jurídico RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"# Assistente em petições jurídicas\n"
        f"LangChain + Chroma + Ollama (`{OLLAMA_MODEL}`) · sessão `{SESSION}`"
    )

    with gr.Tab("Chat RAG"):
        chatbot = gr.Chatbot(height=420, label="Conversa")
        msg = gr.Textbox(label="Mensagem", placeholder="Ex.: Quais tópicos faltam nesta petição?")
        with gr.Row():
            send = gr.Button("Enviar", variant="primary")
            clear = gr.Button("Limpar memória")
        send.click(respond, [msg, chatbot], [chatbot, msg])
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        clear.click(do_clear, outputs=chatbot)

    with gr.Tab("Documentos"):
        gr.Markdown(
            "Coloque PDF/TXT/DOCX em `data/knowledge/` ou envie abaixo. "
            "Depois clique em **Indexar pasta**."
        )
        up = gr.File(label="Upload para data/uploads/")
        up_btn = gr.Button("Enviar e indexar")
        up_out = gr.Textbox(label="Resultado upload")
        up_btn.click(do_upload, up, up_out)
        idx_btn = gr.Button("Indexar knowledge + exemplos JSON")
        idx_out = gr.Textbox(label="Resultado indexação")
        idx_btn.click(do_index_all, outputs=idx_out)

    with gr.Tab("Classificar petição"):
        pet = gr.Textbox(lines=12, label="Texto da petição")
        cls_btn = gr.Button("Classificar (RUIM / BOM / MEIO CERTO)")
        cls_out = gr.Markdown()
        cls_btn.click(do_classify, pet, cls_out)

    with gr.Tab("Refinamento"):
        gr.Markdown("Marque respostas úteis para reforçar o RAG (coleção de refinamento).")
        fq = gr.Textbox(label="Pergunta")
        fr = gr.Textbox(label="Resposta do assistente")
        fc = gr.Textbox(label="Correcção (opcional)")
        fu = gr.Checkbox(label="Resposta útil", value=True)
        fb_btn = gr.Button("Guardar feedback")
        fb_out = gr.Textbox(label="Estado")
        fb_btn.click(do_feedback, [fq, fr, fu, fc], fb_out)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=GRADIO_PORT, share=False)
