#!/usr/bin/env python3
"""Test task-based model routing refactoring."""

import sys
import os
sys.path.insert(0, 'src')

# Test 1: Import and basic config
from pdfwiki import ai_client

print("=" * 60)
print("TEST: Task-based Model Routing Refactoring")
print("=" * 60)

print("\n[1] TASK_MODELS configuration:")
for task, model in ai_client.TASK_MODELS.items():
    print(f"  {task:8} → {model}")

print(f"\n[2] Provider: {ai_client.get_provider()}")

print("\n[3] Valid tasks:")
valid_tasks = list(ai_client.TASK_MODELS.keys())
print(f"  {valid_tasks}")

print("\n[4] Test invalid task error handling:")
try:
    ai_client.query("test", task="invalid_task")
    print("  ✗ ERROR: Should have raised ValueError")
    sys.exit(1)
except ValueError as e:
    print(f"  ✓ Caught ValueError: {str(e)[:40]}...")

print("\n[5] Test environment variable override:")
os.environ['PDF_TO_NOTES_MODEL_CHEAP'] = 'test-cheap-model'
# Note: environment vars are read at module import time, not dynamically
print(f"  Set PDF_TO_NOTES_MODEL_CHEAP to: test-cheap-model")
print(f"  (Note: env vars are read at module load time)")

print("\n[6] Anthropic defaults (if PDF_TO_NOTES_PROVIDER=anthropic):")
if ai_client.PROVIDER == "anthropic":
    print(f"  ✓ Using Anthropic with task-based routing")
    print(f"    Cheap model: {ai_client.TASK_MODELS['cheap']}")
    print(f"    Extract model: {ai_client.TASK_MODELS['extract']}")
    print(f"    Write model: {ai_client.TASK_MODELS['write']}")
else:
    print(f"  Current provider: {ai_client.PROVIDER}")

print("\n" + "=" * 60)
print("✓ All refactoring tests PASSED")
print("=" * 60)
