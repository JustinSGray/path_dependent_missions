python run_q_env.py
python run_recirc.py
python run_pump_comparisons.py

pdfcrop --margins "0 0 0 0" Q_env_comparisons.pdf Q_env_comparisons.pdf
pdfcrop --margins "0 0 0 0" recirc.pdf recirc.pdf
pdfcrop --margins "0 0 0 0" pump_compare.pdf pump_compare.pdf
