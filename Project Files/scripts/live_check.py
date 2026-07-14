import urllib.request, json, urllib.parse

# Test home page
r = urllib.request.urlopen('http://127.0.0.1:5000/')
print(f'HOME   : HTTP {r.status}  ({len(r.read())} bytes)')

# Test health endpoint
r2   = urllib.request.urlopen('http://127.0.0.1:5000/health')
data = json.loads(r2.read())
status_val = data.get('status', '?')
print(f'HEALTH : HTTP {r2.status}  status={status_val}')

# Test predict POST with Rice profile
payload = urllib.parse.urlencode({
    'nitrogen':'90','phosphorus':'42','potassium':'43',
    'temperature':'20.8','humidity':'82.0','ph':'6.5','rainfall':'202.9'
}).encode()
req = urllib.request.Request('http://127.0.0.1:5000/predict', data=payload)
r3  = urllib.request.urlopen(req)
d   = json.loads(r3.read())

print(f'PREDICT: HTTP {r3.status}')
print(f'  success    = {d.get("success")}')
print(f'  crop       = {d.get("crop")}')
print(f'  confidence = {d.get("confidence")}%')
print(f'  season     = {d.get("season")}')
print(f'  soil_type  = {d.get("soil_type")}')
print(f'  irrigation = {d.get("irrigation")}')
print(f'  fertilizer = {d.get("fertilizer")}')
print(f'  yield_exp  = {d.get("yield_exp")}')
print(f'  why (60ch) = {str(d.get("why",""))[:60]}...')
print(f'  advice     = {str(d.get("advice",""))[:60]}...')
print(f'  fi_bars    = {len(d.get("feature_importance", []))} features')
print(f'  probs      = {len(d.get("probabilities", []))} classes')
print(f'  pred_time  = {d.get("pred_time_ms")} ms')

top = sorted(d.get('feature_importance', []), key=lambda x: x['pct'], reverse=True)[:3]
print('  Top 3 FI  :', [(f['label'], f['pct']) for f in top])
