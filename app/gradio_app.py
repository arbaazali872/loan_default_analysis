import gradio as gr
import requests

FLASK_API = "http://localhost:5001/predict"

def predict_loan_default(
    disbursed_amount, asset_cost, ltv,
    branch_id, supplier_id, manufacturer_id, current_pincode_id,
    date_of_birth, employment_type, state_id, employee_code_id,
    mobileno_avl_flag, aadhar_flag, pan_flag, voterid_flag, driving_flag, passport_flag,
    pri_no_of_accts, pri_active_accts, pri_overdue_accts,
    pri_current_balance, pri_sanctioned_amount, pri_disbursed_amount,
    sec_no_of_accts, sec_active_accts, sec_overdue_accts,
    sec_current_balance, sec_sanctioned_amount, sec_disbursed_amount,
    primary_instal_amt, sec_instal_amt,
    new_accts_in_last_six_months, delinquent_accts_in_last_six_months,
    average_acct_age, credit_history_length, no_of_inquiries,
    perform_cns_score, perform_cns_score_description
):
    """Call Flask API and return prediction"""
    
    payload = {
        "disbursed_amount": disbursed_amount,
        "asset_cost": asset_cost,
        "ltv": ltv,
        "branch_id": branch_id,
        "supplier_id": supplier_id,
        "manufacturer_id": manufacturer_id,
        "current_pincode_id": current_pincode_id,
        "date_of_birth": date_of_birth,
        "employment_type": employment_type,
        "state_id": state_id,
        "employee_code_id": employee_code_id,
        "mobileno_avl_flag": mobileno_avl_flag,
        "aadhar_flag": aadhar_flag,
        "pan_flag": pan_flag,
        "voterid_flag": voterid_flag,
        "driving_flag": driving_flag,
        "passport_flag": passport_flag,
        "pri_no_of_accts": pri_no_of_accts,
        "pri_active_accts": pri_active_accts,
        "pri_overdue_accts": pri_overdue_accts,
        "pri_current_balance": pri_current_balance,
        "pri_sanctioned_amount": pri_sanctioned_amount,
        "pri_disbursed_amount": pri_disbursed_amount,
        "sec_no_of_accts": sec_no_of_accts,
        "sec_active_accts": sec_active_accts,
        "sec_overdue_accts": sec_overdue_accts,
        "sec_current_balance": sec_current_balance,
        "sec_sanctioned_amount": sec_sanctioned_amount,
        "sec_disbursed_amount": sec_disbursed_amount,
        "primary_instal_amt": primary_instal_amt,
        "sec_instal_amt": sec_instal_amt,
        "new_accts_in_last_six_months": new_accts_in_last_six_months,
        "delinquent_accts_in_last_six_months": delinquent_accts_in_last_six_months,
        "average_acct_age": average_acct_age,
        "credit_history_length": credit_history_length,
        "no_of_inquiries": no_of_inquiries,
        "perform_cns_score": perform_cns_score,
        "perform_cns_score_description": perform_cns_score_description
    }
    
    try:
        response = requests.post(FLASK_API, json=payload)
        result = response.json()
        
        if response.status_code == 200:
            prediction = "DEFAULT" if result['prediction'] == 1 else "NO DEFAULT"
            prob_no_default = result['probability_no_default'] * 100
            prob_default = result['probability_default'] * 100
            
            return (
                f"**Prediction:** {prediction}\n\n"
                f"**Probability of No Default:** {prob_no_default:.2f}%\n\n"
                f"**Probability of Default:** {prob_default:.2f}%"
            )
        else:
            return f"Error: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Connection Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Loan Default Prediction") as demo:
    gr.Markdown("# Loan Default Prediction System")
    gr.Markdown("Enter loan application details to predict default probability")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Loan Details")
            disbursed_amount = gr.Number(label="Disbursed Amount", value=95000)
            asset_cost = gr.Number(label="Asset Cost", value=100000)
            ltv = gr.Number(label="LTV Ratio", value=95.0)
            
            gr.Markdown("### IDs")
            branch_id = gr.Number(label="Branch ID", value=123)
            supplier_id = gr.Number(label="Supplier ID", value=456)
            manufacturer_id = gr.Number(label="Manufacturer ID", value=789)
            current_pincode_id = gr.Number(label="Pincode ID", value=110001)
            state_id = gr.Number(label="State ID", value=10)
            employee_code_id = gr.Number(label="Employee Code ID", value=555)
            
        with gr.Column():
            gr.Markdown("### Personal Info")
            date_of_birth = gr.Textbox(label="Date of Birth (DD/MM/YYYY)", value="15/06/1990")
            employment_type = gr.Textbox(label="Employment Type", value="Salaried")
            
            gr.Markdown("### Verification Flags")
            mobileno_avl_flag = gr.Number(label="Mobile Available", value=1)
            aadhar_flag = gr.Number(label="Aadhar Flag", value=1)
            pan_flag = gr.Number(label="PAN Flag", value=1)
            voterid_flag = gr.Number(label="Voter ID Flag", value=0)
            driving_flag = gr.Number(label="Driving License Flag", value=1)
            passport_flag = gr.Number(label="Passport Flag", value=0)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Primary Accounts")
            pri_no_of_accts = gr.Number(label="Number of Accounts", value=5)
            pri_active_accts = gr.Number(label="Active Accounts", value=3)
            pri_overdue_accts = gr.Number(label="Overdue Accounts", value=0)
            pri_current_balance = gr.Number(label="Current Balance", value=50000)
            pri_sanctioned_amount = gr.Number(label="Sanctioned Amount", value=200000)
            pri_disbursed_amount = gr.Number(label="Disbursed Amount", value=180000)
            
        with gr.Column():
            gr.Markdown("### Secondary Accounts")
            sec_no_of_accts = gr.Number(label="Number of Accounts", value=2)
            sec_active_accts = gr.Number(label="Active Accounts", value=1)
            sec_overdue_accts = gr.Number(label="Overdue Accounts", value=0)
            sec_current_balance = gr.Number(label="Current Balance", value=20000)
            sec_sanctioned_amount = gr.Number(label="Sanctioned Amount", value=50000)
            sec_disbursed_amount = gr.Number(label="Disbursed Amount", value=45000)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Installments")
            primary_instal_amt = gr.Number(label="Primary Installment", value=5000)
            sec_instal_amt = gr.Number(label="Secondary Installment", value=1500)
            
            gr.Markdown("### Recent Activity")
            new_accts_in_last_six_months = gr.Number(label="New Accounts (6 months)", value=1)
            delinquent_accts_in_last_six_months = gr.Number(label="Delinquent Accounts (6 months)", value=0)
            
        with gr.Column():
            gr.Markdown("### Credit History")
            average_acct_age = gr.Textbox(label="Average Account Age", value="3yrs 6mon")
            credit_history_length = gr.Textbox(label="Credit History Length", value="5yrs 2mon")
            no_of_inquiries = gr.Number(label="Number of Inquiries", value=2)
            
            gr.Markdown("### Credit Score")
            perform_cns_score = gr.Number(label="Performance Score", value=750.0)
            perform_cns_score_description = gr.Textbox(label="Score Description", value="Good")
    
    predict_btn = gr.Button("Predict Default", variant="primary")
    output = gr.Markdown(label="Prediction Result")
    
    predict_btn.click(
        fn=predict_loan_default,
        inputs=[
            disbursed_amount, asset_cost, ltv,
            branch_id, supplier_id, manufacturer_id, current_pincode_id,
            date_of_birth, employment_type, state_id, employee_code_id,
            mobileno_avl_flag, aadhar_flag, pan_flag, voterid_flag, driving_flag, passport_flag,
            pri_no_of_accts, pri_active_accts, pri_overdue_accts,
            pri_current_balance, pri_sanctioned_amount, pri_disbursed_amount,
            sec_no_of_accts, sec_active_accts, sec_overdue_accts,
            sec_current_balance, sec_sanctioned_amount, sec_disbursed_amount,
            primary_instal_amt, sec_instal_amt,
            new_accts_in_last_six_months, delinquent_accts_in_last_six_months,
            average_acct_age, credit_history_length, no_of_inquiries,
            perform_cns_score, perform_cns_score_description
        ],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)