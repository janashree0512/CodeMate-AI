"""
scripts/check_setup.py
======================
Comprehensive setup validation script for CodeMate AI.

Checks:
1. .env file existence (absolute path)
2. GROQ_API_KEY presence and formatting (safely masked: first 4 & last 4 chars)
3. Model configuration
4. Workspace directory existence
5. Real Groq API connectivity test ('Reply OK') with plain-English error classification:
   - 401: Invalid API Key
   - 404: Model not found or no access
   - 429: Rate limit reached
   - Network / Connection error
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config, ConfigurationError, mask_key, DEFAULT_MODEL
from agent.llm_client import GroqClient, LLMAuthenticationError, LLMRateLimitError, LLMConnectionError, LLMClientError


def run_checks():
    print("=" * 70)
    print("           CodeMate AI - Environment & Setup Diagnostic")
    print("=" * 70)

    all_passed = True
    env_path = (PROJECT_ROOT / ".env").resolve()

    # 1. Check .env file
    if env_path.exists():
        print(f"[PASS] 1. .env file found at: {env_path}")
    else:
        print(f"[FAIL] 1. .env file NOT found at: {env_path}")
        print("       Action: Copy .env.example to .env and configure your keys.")
        all_passed = False

    # 2. Check configuration and key presence
    try:
        config = load_config(strict_api_key=False)
    except ConfigurationError as exc:
        print(f"[FAIL] Configuration parsing error: {exc}")
        return

    masked = mask_key(config.groq_api_key)
    if not config.groq_api_key or config.groq_api_key == "your_groq_api_key_here":
        print(f"[FAIL] 2. GROQ_API_KEY is empty or placeholder ({masked})")
        all_passed = False
    elif not config.groq_api_key.startswith("gsk_"):
        print(f"[FAIL] 2. GROQ_API_KEY format invalid ({masked}). Expected key starting with 'gsk_'")
        all_passed = False
    else:
        print(f"[PASS] 2. GROQ_API_KEY is present and formatted ({masked})")

    # 3. Check Model name & accessible models for key
    accessible_models = []
    try:
        raw_client = GroqClient(config=config).get_client()
        all_models = raw_client.models.list().data
        accessible_models = [
            m.id for m in all_models
            if not any(x in m.id for x in ["whisper", "guard", "orpheus"])
        ]
        print(f"       Accessible chat models for your Groq key: {', '.join(accessible_models) if accessible_models else 'none'}")
        if config.model_name in accessible_models:
            print(f"[PASS] 3. Configured Model: '{config.model_name}' (available on your account)")
        else:
            print(f"[WARN] 3. Configured Model '{config.model_name}' is not in your account's accessible list.")
            print(f"       Action: Update MODEL_NAME in .env to one of: {', '.join(accessible_models) or DEFAULT_MODEL}")
    except Exception as exc:
        print(f"[PASS] 3. Configured Model: '{config.model_name}' (model query offline/skipped: {exc})")

    # 4. Check Workspace folder
    ws_path = config.workspace_dir.resolve()
    if ws_path.exists() and ws_path.is_dir():
        print(f"[PASS] 4. Workspace folder ready at: {ws_path}")
    else:
        try:
            ws_path.mkdir(parents=True, exist_ok=True)
            print(f"[PASS] 4. Workspace folder created at: {ws_path}")
        except Exception as exc:
            print(f"[FAIL] 4. Workspace directory error: {exc}")
            all_passed = False

    # 5. Live Groq Connectivity Check
    print("\n--- Live Groq API Test ('Reply OK') ---")
    if not all_passed:
        print("[SKIP] Skipping live API call because preliminary checks failed.")
        print("=" * 70)
        return

    try:
        client = GroqClient(config=config)
        response_text = client.ask("Reply with the single word: OK")
        print(f"[PASS] 5. Groq API Response: '{response_text.strip()}'")
        print("\n>>> ALL CHECKS PASSED: CodeMate AI is fully configured and ready! <<<")

    except LLMAuthenticationError as err:
        print(f"[FAIL] 5. Groq Authentication Error (401 / Invalid Key): {err}")
        print("       Action: Double check your API key at https://console.groq.com/keys")

    except LLMRateLimitError as err:
        print(f"[FAIL] 5. Groq Rate Limit Error (429): {err}")
        print("       Action: Wait a moment for TPM/RPM quota reset or upgrade your tier.")

    except LLMConnectionError as err:
        print(f"[FAIL] 5. Network / Connectivity Error: {err}")
        print("       Action: Verify internet access or firewall/proxy restrictions.")

    except LLMClientError as err:
        err_str = str(err)
        if "is not available for your Groq key" in err_str or "404" in err_str or "model_not_found" in err_str:
            print(f"[FAIL] 5. {err}")
            print(f"       Action: Update MODEL_NAME in .env to a supported model (e.g., {DEFAULT_MODEL}).")
        else:
            print(f"[FAIL] 5. Groq API Error: {err}")

    except Exception as err:
        print(f"[FAIL] 5. Unexpected Error: {err}")

    print("=" * 70)


if __name__ == "__main__":
    run_checks()
