

from typing import List, Dict, Any
import cvxpy as cp




def print_summary(summary_data: List[Dict[str, Any]], problem: cp.Problem) -> None:
    """
    Print a formatted summary of the fitted model.
    """
    total_params = sum(item["Parameters"] for item in summary_data)
    status = problem.status if problem else "Not Fitted"
    
    width = 120
    print("\n" + "="*width)
    print("✨ Model Summary ✨")
    print("="*width)
    status_icon = f"✅ {status}" if status == "optimal" else f"❌ {status}"
    print(f"Problem Status: {status_icon}")
    print("-" * width)
    print(f"\033[1m{'Spline Type':<20} | {'Term':<12} | {'Tag':<15} | {'Constraints':<20} | {'Penalties':<20} | {'Params':<8}\033[0m")
    print("-" * width)
    for item in summary_data:
        tag = item.get("Tag")
        tag_str = str(tag) if tag is not None else "None"
        penalties_str = item.get("Penalties", "None")
        print(f"🟢 {item['Spline Type']:<17} | {item['Term']:<12} | {tag_str:<15} | {item['Constraints']:<20} | {penalties_str:<20} | {item['Parameters']:<8}")
    print("-" * width)
    print(f"{'📊 Total Parameters':<98} | {total_params}")
    print("="*width + "\n")


