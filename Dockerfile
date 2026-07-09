# Step 1: Use official stable lightweight Python base image
FROM python:3.10-slim

# Step 2: Set environment variables to prevent Python from buffering outputs
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Step 3: Set up a secure, non-root user account required by cloud platforms
RUN useradd -m -u 1000 user
WORKDIR $HOME/app

# Step 4: Pre-create target server runtime directories
RUN mkdir -p backend frontend

# Step 5: Copy dependency mappings and install packages
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 6: Move codebase structures into container virtual workspace
COPY --chown=user backend/ ./backend/
COPY --chown=user frontend/ ./frontend/

# Step 7: Change execution permissions to user scope
USER user

# Step 8: Open up port 7860 (Hugging Face default listening pipeline target)
EXPOSE 7860

# Step 9: Launch the unified operational server via the backend directory
CMD ["python", "backend/app.py"]