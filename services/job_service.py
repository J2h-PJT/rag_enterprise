from sqlalchemy import text

class JobService:

    def __init__(self, engine):
        self.engine = engine

    def cleanup(self):
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE upload_jobs
                SET status='pending'
                WHERE status='processing'
            """))

    def fetch_next(self):
        with self.engine.begin() as conn:
            row = conn.execute(text("""
                SELECT id, file_id
                FROM upload_jobs
                WHERE status='pending'
                ORDER BY created_at
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            """)).fetchone()

            if not row:
                return None

            conn.execute(text("""
                UPDATE upload_jobs
                SET status='processing'
                WHERE id=:id
            """), {"id": row.id})

            return row

    def mark_done(self, job_id):
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE upload_jobs
                SET status='done'
                WHERE id=:id
            """), {"id": job_id})

    def mark_failed(self, job_id):
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE upload_jobs
                SET status='failed'
                WHERE id=:id
            """), {"id": job_id})
