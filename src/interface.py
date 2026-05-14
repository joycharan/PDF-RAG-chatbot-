import gradio as gr

# Gradio application setup
def create_demo():
    with gr.Blocks(
        title="RAG Chatbot Q&A",
        theme="Soft",
        analytics_enabled=False,  # Disable analytics to prevent session tracking issues
        css=".gradio-container {max-width: 90% !important}"  # Better responsive layout
    ) as demo:
        # Add a state component to store the current session state
        session_state = gr.State({})
        
        with gr.Column():
            # Add a header
            gr.Markdown("# PDF Question Answering with RAG")
            gr.Markdown("Upload a PDF and ask questions about its content.")
            
            with gr.Row():
                # Chat history on left, PDF viewer on right
                chat_history = gr.Chatbot(
                    value=[],
                    elem_id='chatbot',
                    height=680,
                    show_label=False
                )
                
                show_img = gr.Image(
                    label='PDF Preview',
                    height=680,
                    show_label=True
                )

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    show_label=False,
                    placeholder="Type here to ask your PDF",
                    container=False,
                    autofocus=True
                )

            with gr.Column(scale=1):
                submit_button = gr.Button('Send', variant="primary")

            with gr.Column(scale=1):
                uploaded_pdf = gr.UploadButton(
                    "📁 Upload PDF",
                    file_types=[".pdf"],
                    variant="secondary"
                )
        
        # Add examples section (optional)
        gr.Examples(
            examples=[
                ["What is this document about?"],
                ["Can you summarize this PDF?"],
                ["What are the main points in this document?"]
            ],
            inputs=text_input
        )
        
        # Add footer with version info
        gr.Markdown("### RAG PDF Chatbot v1.0")

        return demo, chat_history, show_img, text_input, submit_button, uploaded_pdf

if __name__ == '__main__':
    demo, chatbot, show_img, text_input, submit_button, uploaded_pdf = create_demo()
    demo.queue(concurrency_count=1)  # Ensure sequential processing
    demo.launch()
