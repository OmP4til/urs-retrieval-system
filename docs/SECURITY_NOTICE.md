# Security Notice

## Database Password Exposure

⚠️ **IMPORTANT**: A database password was accidentally committed to git history in commit `d7729c0`.

### Immediate Actions Required:

1. **Change the PostgreSQL password immediately:**
   ```sql
   ALTER USER postgres WITH PASSWORD 'your_new_secure_password';
   ```

2. **Update your local .env file:**
   ```
   POSTGRES_PASSWORD=your_new_secure_password
   ```

3. **Never commit sensitive data:**
   - Always use environment variables for credentials
   - Check `.env` is in `.gitignore`
   - Review commits before pushing

### Current Security Measures:

✅ Hardcoded password removed from code (commit `c7fd393`)
✅ Using environment variables for all database credentials
✅ `.env.example` provided as template
✅ `__pycache__` files removed from version control

### Best Practices Going Forward:

1. **Always use environment variables for:**
   - API keys
   - Database passwords
   - Any sensitive configuration

2. **Before committing, check for secrets:**
   ```powershell
   git diff --cached | Select-String -Pattern "(password|api[_-]?key|secret|token)"
   ```

3. **Use git-secrets or similar tools:**
   ```powershell
   # Install git-secrets to prevent committing secrets
   # https://github.com/awslabs/git-secrets
   ```

### Environment Variable Configuration:

The application now reads database credentials from environment variables:

- `POSTGRES_HOST` (default: localhost)
- `POSTGRES_PORT` (default: 5433)
- `POSTGRES_DB` (default: urs_gemini)
- `POSTGRES_USER` (default: postgres)
- `POSTGRES_PASSWORD` (REQUIRED - no default for security)

Copy `.env.example` to `.env` and update with your actual credentials.
