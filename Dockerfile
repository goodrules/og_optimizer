# Use the pre-built multi-arch NiceGUI image
FROM zauberzeug/nicegui:latest

# Set working directory to /app
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (including Chromium)
#RUN playwright install chromium

# Copy the rest of the application code
COPY . /app/

# Expose port 8080 on the container
EXPOSE 8080

# Set the same environment variable at runtime
#ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Command to run the application
CMD ["python", "main.py"]