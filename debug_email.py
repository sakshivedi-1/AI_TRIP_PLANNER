import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st

# HARDCODE YOUR CREDENTIALS HERE FOR THIS TEST ONLY
# (Delete this file after testing)
TEST_EMAIL = "your_mail"
TEST_PASSWORD = "APP_PASSWORD"  # Use an App Password, not your regular password

def run_test():
    print("--- STARTING EMAIL TEST ---")
    
    try:
        # 1. Test Connection to Gmail
        print(f"1. Attempting to connect to smtp.gmail.com:587...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        print("   Connection successful!")
        
        # 2. Test Security Handshake
        print("2. Starting TLS encryption...")
        server.starttls()
        print("   TLS successful!")
        
        # 3. Test Login
        print(f"3. Logging in as {TEST_EMAIL}...")
        server.login(TEST_EMAIL, TEST_PASSWORD)
        print("   Login successful!")
        
        # 4. Test Sending
        print("4. Sending test email...")
        msg = MIMEMultipart()
        msg['From'] = TEST_EMAIL
        msg['To'] = TEST_EMAIL  # Send to yourself
        msg['Subject'] = "DEBUG TEST: If you see this, it works!"
        msg.attach(MIMEText("This is a test email from Python.", 'plain'))
        
        server.sendmail(TEST_EMAIL, TEST_EMAIL, msg.as_string())
        print("   Email command sent to server!")
        
        server.quit()
        print("--- TEST COMPLETED SUCCESSFULLY ---")
        print("CHECK YOUR INBOX (AND SPAM FOLDER) NOW.")
        
    except smtplib.SMTPAuthenticationError:
        print("\n❌ AUTH ERROR: Password rejected.")
        print("   - check if you pasted the 16-char code correctly (no extra spaces)")
        print("   - check if the email address is 100% correct")
    except TimeoutError:
        print("\n❌ TIMEOUT ERROR: Connection blocked.")
        print("   - Your Wi-Fi is blocking port 587.")
        print("   - Try switching to a Mobile Hotspot and try again.")
    except Exception as e:
        print(f"\n❌ OTHER ERROR: {e}")

if __name__ == "__main__":
    run_test()