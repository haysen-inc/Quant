import re
import json

def parse_mylanguage(raw_code: str) -> dict:
    """
    Unified entry point to parse a raw MyLanguage script into initialized constants.
    Returns default values if parsing fails.
    """
    config = {
        'w_bias': -50.0,
        'w_f1': 6.0,
        'w_f2': 6.0,
        'w_cond3_j1': 1.0
    }
    
    jx_res = MyLanguageParser.parse_jx_formula(raw_code)
    if jx_res and isinstance(jx_res, dict) and jx_res.get('success'):
        config.update(jx_res['parsed_constants'])
        
    bk_res = MyLanguageParser.parse_bk2_cond3(raw_code)
    if bk_res and isinstance(bk_res, dict) and bk_res.get('success'):
        config['w_cond3_j1'] = bk_res['w_cond3_j1']
        
    return config

class MyLanguageParser:
    """
    Parses a constrained subset of MyLanguage syntax (specifically the JX component logic) 
    into variable names and literal coefficient strings intended for PyTorch initialization.
    """
    
    @staticmethod
    def parse_jx_formula(formula_str):
        """
        Input: "JX:=J1+J2-50+J2*TEMA3T3*6+J1*TEMA3T2*6+J3*TEMA3T1;" or "JX:..."
        Output: Extracts the biases and weights.
        """
        # Clean string
        formula = formula_str.replace(" ", "").replace("\n", "")
        
        # Robustly search for the JX formula amidst multiline text
        # Make the '=' optional since standard indicators use ':'
        match = re.search(r'JX:=?(.*?)[;,]', formula)
        if not match:
            return {"success": False, "error": "Formula must contain a line starting with JX: or JX:="}
            
        right_side = match.group(1)
        
        # Super simplified heuristic parser for the exact expected format
        # J1+J2[-BIAS]+J2*TEMA3T3*[W_F1]+J1*TEMA3T2*[W_F2]+J3*TEMA3T1
        
        # We use regex to find standalone numbers (bias) and multipliers (*N)
        # 1. Find Bias (a number preceded by + or - and not followed by a *)
        bias_match = re.search(r'([+-]\d+(?:\.\d+)?)(?![*a-zA-Z])', right_side)
        w_bias_val = float(bias_match.group(1)) if bias_match else 0.0
        
        # Helper to extract either a digit map or look up a variable like NP1
        def extract_multiplier(keyword, text):
            m = re.search(f'{keyword}\\*([a-zA-Z0-9_.]+)', right_side)
            if m:
                val_str = m.group(1)
                if val_str.replace('.', '', 1).isdigit():
                    return float(val_str)
                # It's a variable like NP1, find its definition
                var_m = re.search(f"{val_str}:=(\\d+(?:\\.\\d+)?)", formula_str.replace(" ", ""))
                if var_m:
                    return float(var_m.group(1))
            return 1.0 # Standard multiplier default

        w_f1_val = extract_multiplier("TEMA3T3", right_side)
        w_f2_val = extract_multiplier("TEMA3T2", right_side)
        
        return {
            "success": True,
            "parsed_constants": {
                "w_bias": w_bias_val,
                "w_f1": w_f1_val,
                "w_f2": w_f2_val
            }
        }
        
    @staticmethod
    def parse_bk2_cond3(bk2_str):
        """
        Input: "BK2:= ... AND (JX > J1 * 1.5)"
        Extracts the multiplier on J1.
        """
        clean = bk2_str.replace(" ", "")
        
        # Look for JX>J1 or JX>J1*N
        match = re.search(r'JX>J1(?:(?:\*|/)([+-]?\d+(?:\.\d+)?))?', clean)
        if match:
            # If there's a multiplier group, extract it, else it's 1.0
            mult = float(match.group(1)) if match.group(1) else 1.0
            return {"success": True, "w_cond3_j1": mult}
            
        return {"success": False, "error": "Could not locate JX>J1 condition."}

if __name__ == "__main__":
    test_str = "JX:=J1+J2-50.5+J2*TEMA3T3*6.1+J1*TEMA3T2*6.2+J3*TEMA3T1;"
    print("Test JX:", MyLanguageParser.parse_jx_formula(test_str))
    
    test_bk2 = "BK2:= (REF(J1,1)<REF(J2,1)) AND (J1>J2) AND C>MA_C_DOWN AND (JX > J1 * 1.0);"
    print("Test BK:", MyLanguageParser.parse_bk2_cond3(test_bk2))
