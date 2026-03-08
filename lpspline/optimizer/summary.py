

from typing import List, Dict, Any
import cvxpy as cp




def print_summary(summary_data: List[Dict[str, Any]], status: str = None) -> None:
    """
    Print a formatted summary console output of the fitted regression model characteristics.

    Parameters
    ----------
    summary_data : List[Dict[str, Any]]
        A list of mapped dictionaries detailing the specific features per Spline component.
    status: str
        Problem status
    """
    total_params = sum(item["Parameters"] for item in summary_data)
    status = status if status is not None else "Not fitted"
    
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


