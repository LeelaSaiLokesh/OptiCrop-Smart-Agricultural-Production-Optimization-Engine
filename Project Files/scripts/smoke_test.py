"""
smoke_test.py — OptiCrop live prediction smoke test
Run: python smoke_test.py
"""
import sys, json
sys.path.insert(0, r'e:\OptiCrop – Smart Agricultural Production Optimization Engine')

from app import create_app

app = create_app('testing')

PROFILES = [
    ("Rice",   {'nitrogen':'90','phosphorus':'42','potassium':'43',
                'temperature':'20.8','humidity':'82.0','ph':'6.5','rainfall':'202.9'}),
    ("Cotton", {'nitrogen':'118','phosphorus':'46','potassium':'20',
                'temperature':'23.9','humidity':'79.4','ph':'6.86','rainfall':'51.8'}),
    ("Maize",  {'nitrogen':'75','phosphorus':'68','potassium':'98',
                'temperature':'22.6','humidity':'62.0','ph':'6.1','rainfall':'89.5'}),
]

print("=" * 60)
print("  OptiCrop — Live Prediction Smoke Test")
print("=" * 60)

all_ok = True
for expected, payload in PROFILES:
    with app.test_client() as c:
        r = c.post('/predict', data=payload)
        d = json.loads(r.data)
        crop      = d.get('crop', 'N/A')
        conf      = d.get('confidence', 0)
        fi_count  = len(d.get('feature_importance', []))
        top_feat  = sorted(d.get('feature_importance', []),
                           key=lambda x: x['pct'], reverse=True)
        top_label = top_feat[0]['label'] if top_feat else 'N/A'
        top_pct   = top_feat[0]['pct']   if top_feat else 0
        status    = "PASS" if d.get('success') else "FAIL"
        if not d.get('success'):
            all_ok = False

        print(f"\n  Profile   : {expected}")
        print(f"  Predicted : {crop}  ({conf}% confidence)")
        print(f"  Top feat  : {top_label} = {top_pct}%")
        print(f"  FI bars   : {fi_count}")
        print(f"  HTTP      : {r.status_code}  [{status}]")

print()
print("=" * 60)
if all_ok:
    print("  ALL SMOKE TESTS PASSED")
else:
    print("  SOME TESTS FAILED — review above")
print("=" * 60)
