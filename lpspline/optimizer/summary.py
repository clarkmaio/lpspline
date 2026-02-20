

from typing import List, Dict, Any
import cvxpy as cp




def print_summary(summary_data: List[Dict[str, Any]], problem: cp.Problem) -> None:
    """
    Print a formatted summary of the fitted model.
    """
    total_params = sum(item["Parameters"] for item in summary_data)
    status = problem.status if problem else "Not Fitted"
    
    print("\n" + "="*80)
    print("✨ Model Summary ✨")
    print("="*80)
    print(f"Problem Status: " + f"✅ {status}" if status == "optimal" else f"❌ {status}")
    print("-" * 80)
    print(f"{'Spline Type':<25} | {'Term':<15} | {'Params':<8} | {'Constraints':<30}")
    print("-" * 80)
    for item in summary_data:
        print(f"🟢 {item['Spline Type']:<22} | {item['Term']:<15} | {item['Parameters']:<8} | {item['Constraints']:<30}")
    print("-" * 80)
    print(f"{'📊 Total Parameters':<42} | {total_params:<8} |")
    print("="*80 + "\n")


