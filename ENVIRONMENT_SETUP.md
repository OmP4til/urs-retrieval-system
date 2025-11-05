# Environment Variables Setup

## Important Security Notice
Never commit API keys or sensitive credentials to git repositories. This project uses environment variables to manage sensitive configuration.

## Setup Instructions

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and add your actual API keys and credentials:
   ```
   # Gemini API Configuration
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   
   # PostgreSQL Database Settings
   DB_HOST=localhost
   DB_PORT=5433
   DB_NAME=requirements_db
   DB_USER=postgres
   DB_PASSWORD=your_database_password
   ```

3. The `.env` file is automatically ignored by git (listed in `.gitignore`)

## Required Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini Pro API key
- Database credentials (if different from defaults)

## Getting a Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file

## Security Best Practices

- Never share your `.env` file
- Never commit API keys to version control
- Use different API keys for development and production
- Regularly rotate your API keys