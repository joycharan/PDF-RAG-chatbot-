from interface import create_demo
from pdfchatbot import PDFChatBot

# Create PDFChatBot instance
pdf_chatbot = PDFChatBot()

# Create Gradio interface
demo, chat_history, show_img, txt, submit_button, uploaded_pdf = create_demo()

# Set up event handlers
with demo:
    # Event handler for uploading a PDF
    uploaded_pdf.upload(pdf_chatbot.render_file, inputs=[uploaded_pdf], outputs=[show_img])

    # Event handler for submitting text and generating response
    # Use a single chain for better session management
    submit_button.click(
        fn=pdf_chatbot.chat_flow,
        inputs=[chat_history, txt, uploaded_pdf],
        outputs=[chat_history, txt, show_img],
        queue=True  # Enable queueing for all operations
    )

if __name__ == "__main__":
    # Just enable queue without deprecated parameters
    demo.queue()
    # Set max_threads to control total number of workers
    demo.launch(share=False, max_threads=1)  # Launch with sharing disabled