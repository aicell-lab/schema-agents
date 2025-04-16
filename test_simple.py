#!/usr/bin/env python
# coding=utf-8

from src.schema_agents.local_python_executor import LocalPythonExecutor, fix_final_answer_code

# Test fix_final_answer_code function
print("Testing fix_final_answer_code function:")
code1 = "final_answer = 'This is my answer'"
print(f"Original: {code1}")
fixed1 = fix_final_answer_code(code1)
print(f"Fixed: {fixed1}")

code2 = "final_answer('This is my answer')"
print(f"\nOriginal: {code2}")
fixed2 = fix_final_answer_code(code2)
print(f"Fixed: {fixed2}")

code3 = "final_answer = 'Answer 1'\nfinal_answer('Answer 2')"
print(f"\nOriginal: {code3}")
fixed3 = fix_final_answer_code(code3)
print(f"Fixed: {fixed3}")

print("\nIf you see this message, the function syntax is correct!")
print("This confirms that our indentation fixes successfully resolved the syntax issues.") 