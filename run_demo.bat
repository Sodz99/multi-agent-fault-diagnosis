@echo off
cd /d "C:\Users\House\OneDrive\Desktop\Sohan_Arun\Career\Portfolio Projects\Fault Diagnosis Multi Agent"
venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); from fault_diagnosis.cli import main; main(['fault-diagnosis', '--session', 'demo'])"
pause