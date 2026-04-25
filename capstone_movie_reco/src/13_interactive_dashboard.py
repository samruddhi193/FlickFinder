
import os
html_content = '''
<html><body><h1>Interactive Dashboard</h1><p>Mock capstone dashboard.</p></body></html>
'''
os.makedirs('outputs', exist_ok=True)
with open('outputs/interactive_dashboard.html', 'w') as f:
    f.write(html_content)
    
print("Dashboard created.")
