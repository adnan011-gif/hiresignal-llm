"""
End-to-end test script for HireSignal JD Analyzer.
Tests file extraction + textbox population for all 3 tabs using gradio_client.
"""
import sys
import time
from gradio_client import Client

APP_URL = "http://127.0.0.1:7860"
PASS = 0
FAIL = 0

def report(test_name, passed, detail=""):
    global PASS, FAIL
    if passed:
        PASS += 1
        print(f"  ✅ {test_name}")
    else:
        FAIL += 1
        print(f"  ❌ {test_name}: {detail}")

print("🚀 Connecting to Gradio app...")
try:
    client = Client(APP_URL)
except Exception as e:
    print(f"❌ Cannot connect to {APP_URL}: {e}")
    sys.exit(1)

print(f"✅ Connected\n")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: JD Analyzer
# ═══════════════════════════════════════════════════════════════════════════
print("━" * 60)
print("TAB 1: JD Analyzer")
print("━" * 60)

# Test 1a: Extract from PDF
print("\n--- Test 1a: Extract JD from PDF ---")
try:
    result = client.predict(
        file="test_files/test_jd.pdf",
        api_name="/extract_file"
    )
    text, status = result
    has_text = bool(text and text.strip())
    report("PDF extraction returns text", has_text,
           f"text='{text[:80]}...' status='{status}'" if has_text else f"EMPTY text, status='{status}'")
    report("Status shows success", "✅" in status, f"status='{status}'")
    if has_text:
        print(f"    Extracted {len(text)} chars: {text[:100]}...")
except Exception as e:
    report("PDF extraction", False, str(e))

# Test 1b: Extract from TXT
print("\n--- Test 1b: Extract JD from TXT ---")
try:
    result = client.predict(
        file="test_files/test_jd.txt",
        api_name="/extract_file"
    )
    text, status = result
    has_text = bool(text and text.strip())
    report("TXT extraction returns text", has_text,
           f"text='{text[:80]}'" if has_text else f"EMPTY text, status='{status}'")
except Exception as e:
    report("TXT extraction", False, str(e))

# Test 1c: Analyze JD (with text input, to verify analyze_jd works)
print("\n--- Test 1c: Analyze JD with text input ---")
try:
    job = client.submit(
        "Senior Backend Engineer. Requirements: 5+ years Python, Django, PostgreSQL, AWS, Docker, CI/CD.",
        api_name="/analyze_jd"
    )
    result = job.result(timeout=120)
    has_result = bool(result and result.strip() and "❌" not in result)
    report("analyze_jd returns real output", has_result,
           f"result='{str(result)[:100]}'" if not has_result else "")
    if has_result:
        print(f"    Analysis ({len(result)} chars): {result[:120]}...")
except Exception as e:
    report("analyze_jd", False, str(e))

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Fit Scorer
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "━" * 60)
print("TAB 2: Fit Scorer")
print("━" * 60)

# Test 2a: Extract JD from PDF (Fit Scorer tab)
print("\n--- Test 2a: Extract JD from PDF (Fit Scorer) ---")
try:
    result = client.predict(
        file="test_files/test_jd.pdf",
        api_name="/extract_file_1"
    )
    text, status = result
    has_text = bool(text and text.strip())
    report("Fit Scorer JD extraction", has_text,
           f"text='{text[:80]}'" if has_text else f"EMPTY, status='{status}'")
except Exception as e:
    # The API name might differ — try without _1
    report("Fit Scorer JD extraction", False, str(e))

# Test 2b: Extract Resume from DOCX (Fit Scorer tab)
print("\n--- Test 2b: Extract Resume from DOCX (Fit Scorer) ---")
try:
    result = client.predict(
        file="test_files/test_resume.docx",
        api_name="/extract_file_2"
    )
    text, status = result
    has_text = bool(text and text.strip())
    report("Fit Scorer Resume DOCX extraction", has_text,
           f"text='{text[:80]}'" if has_text else f"EMPTY, status='{status}'")
except Exception as e:
    report("Fit Scorer Resume DOCX extraction", False, str(e))

# Test 2c: Score Fit with text inputs
print("\n--- Test 2c: Score Fit with text inputs ---")
try:
    job = client.submit(
        "Backend Engineer. 3+ years Python, PostgreSQL, REST APIs, AWS.",
        "2 years Python, built REST APIs with Flask, used PostgreSQL.",
        api_name="/score_fit"
    )
    result = job.result(timeout=120)
    has_result = bool(result and result.strip() and "❌" not in result)
    report("score_fit returns real output", has_result,
           f"result='{str(result)[:100]}'" if not has_result else "")
    if has_result:
        print(f"    Score ({len(result)} chars): {result[:120]}...")
except Exception as e:
    report("score_fit", False, str(e))

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Resume Tip Generator
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "━" * 60)
print("TAB 3: Resume Tip Generator")
print("━" * 60)

# Test 3a: Extract JD from PDF (Resume Tip tab)
print("\n--- Test 3a: Extract JD from PDF (Resume Tip) ---")
try:
    result = client.predict(
        file="test_files/test_jd.pdf",
        api_name="/extract_file_3"
    )
    text, status = result
    has_text = bool(text and text.strip())
    report("Resume Tip JD extraction", has_text,
           f"text='{text[:80]}'" if has_text else f"EMPTY, status='{status}'")
except Exception as e:
    report("Resume Tip JD extraction", False, str(e))

# Test 3b: Extract Resume from DOCX (Resume Tip tab)
print("\n--- Test 3b: Extract Resume from DOCX (Resume Tip) ---")
try:
    result = client.predict(
        file="test_files/test_resume.docx",
        api_name="/extract_file_4"
    )
    text, status = result
    has_text = bool(text and text.strip())
    report("Resume Tip Resume DOCX extraction", has_text,
           f"text='{text[:80]}'" if has_text else f"EMPTY, status='{status}'")
except Exception as e:
    report("Resume Tip Resume DOCX extraction", False, str(e))

# Test 3c: Improve Resume with text inputs
print("\n--- Test 3c: Improve Resume with text inputs ---")
try:
    job = client.submit(
        "Product Manager. Requirements: data-driven, A/B testing, SQL.",
        "- Managed product roadmap\n- Worked with engineering teams\n- Analyzed user feedback",
        api_name="/improve_resume"
    )
    result = job.result(timeout=120)
    has_result = bool(result and result.strip() and "❌" not in result)
    report("improve_resume returns real output", has_result,
           f"result='{str(result)[:100]}'" if not has_result else "")
    if has_result:
        print(f"    Improved ({len(result)} chars): {result[:120]}...")
except Exception as e:
    report("improve_resume", False, str(e))

# ═══════════════════════════════════════════════════════════════════════════
# Test OCR (image extraction)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "━" * 60)
print("BONUS: Image OCR extraction")
print("━" * 60)
print("\n--- Test 4: Extract from PNG (OCR) ---")
try:
    result = client.predict(
        file="test_files/test_ocr.png",
        api_name="/extract_file"
    )
    text, status = result
    has_text = bool(text and text.strip())
    report("PNG OCR extraction returns text", has_text,
           f"text='{text[:80]}'" if has_text else f"EMPTY, status='{status}'")
except Exception as e:
    report("PNG OCR extraction", False, str(e))

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS+FAIL} tests")
print("═" * 60)

if FAIL > 0:
    sys.exit(1)
else:
    print("🎉 All tests passed!")
    sys.exit(0)
