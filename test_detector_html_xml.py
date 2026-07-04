from detector import detect

# Case 1: XHTML with XML prolog — should now be reclassified as 'html'
xhtml_with_prolog = b'''<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Test 10-K</title></head>
<body><table><tr><td>Total net sales</td></tr></table></body>
</html>'''

# Case 2: Genuine XML, no HTML — should stay 'xml'
plain_xml = b'''<?xml version="1.0" encoding="UTF-8"?>
<root><record><field>value</field></record></root>'''

r1 = detect(xhtml_with_prolog, filename="test.xml")
r2 = detect(plain_xml, filename="test.xml")

print(f"XHTML w/ prolog -> parser_key: {r1.parser_key}, mime: {r1.mime_type}")  # expect parser_key='html'
print(f"Plain XML       -> parser_key: {r2.parser_key}, mime: {r2.mime_type}")  # expect parser_key='xml'

assert r1.parser_key == 'html', f"FAIL: expected 'html', got '{r1.parser_key}'"
assert r2.parser_key == 'xml', f"FAIL: expected 'xml', got '{r2.parser_key}'"
print("PASS")