# OzABNxpenseTracker

OzABNxpenseTracker is a simple, self-hosted expense tracking application for Australian sole traders and small businesses. It allows you to record expenses, upload and attach receipts (with OCR), categorize them, and generate reports for tax purposes.

## Features

*   **Expense Tracking:** Record vendor, amount, date, description, and category for each expense.
*   **Receipt OCR:** Upload a receipt image and the application will attempt to automatically extract the vendor, amount, and date.
*   **Recurring Expenses:** Set up and track recurring expenses (e.g., monthly subscriptions).
*   **Reporting:** Generate CSV reports of your expenses for a given date range.
*   **Tax Guide:** Includes a helpful guide on common tax-deductible expenses for Australian sole traders.
*   **Dockerized:** The entire application is containerized for easy deployment.

## Installation

This application is designed to be run with Docker and Docker Compose.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/jasoisjaso/OzABNxpenseTracker.git
    cd OzABNxpenseTracker
    ```

2.  **Build and run the application:**

    ```bash
    docker-compose up -d --build
    ```

3.  **Access the application:**

    Open your web browser and navigate to `http://localhost:80`.

## Deployment to Railway

1.  **Push to GitHub:** Make sure your code is pushed to your GitHub repository.
2.  **Create a Railway Project:** Create a new project in Railway and link it to your GitHub repository.
3.  **Add a Database:** Add a PostgreSQL database service to your project.
4.  **Configure Environment Variables:** Railway will likely detect your `docker-compose.yml` and create the `frontend` and `backend` services. You will need to add the `DATABASE_URL` environment variable to your `backend` service, with the connection string provided by your Railway PostgreSQL service.
5.  **Update Backend Code:** You will need to modify the `main.py` file to use the PostgreSQL database instead of SQLite. I can help with this.
