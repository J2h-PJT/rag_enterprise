class DocumentWorker:
    def __init__(self, core_engine, pdf_processor):
        self.core_engine = core_engine
        self.pdf_processor = pdf_processor

    def process_pdf(self, file_path, file_id):
        self.pdf_processor.process(file_path, file_id)
