import time
from core.db import create_db_engine
from core.vectorstore import create_vectorstore
from services.job_service import JobService
from services.pdf_processor import PDFProcessor

engine = create_db_engine()
vectorstore = create_vectorstore(engine)

job_service = JobService(engine)
pdf_service = PDFProcessor(engine, vectorstore)

def worker_loop():

    print("🚀 Worker started")
    job_service.cleanup()

    while True:

        job = job_service.fetch_next()

        if not job:
            time.sleep(2)
            continue

        try:
            pdf_service.process(job.file_id)
            job_service.mark_done(job.id)
            print("✅ Job done")

        except Exception as e:
            print("❌ Job failed:", e)
            job_service.mark_failed(job.id)

if __name__ == "__main__":
    worker_loop()
