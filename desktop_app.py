import os
import re
import json
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from typing import List, Dict
import threading

from agent import fetch_recent_emails, process_email
from emailapi import get_service, send_email, get_user_email
from jarvis import JARVISAssistant, Command, CommandType
from datetime import datetime


class EmailAgentDesktop:
    def __init__(self, root):
        self.root = root
        self.root.title("📬 Email Agent Desktop")
        self.root.geometry("1200x800")
        
        # State variables
        self.connected = False
        self.emails = []
        self.results = []
        self.auto_sent_ids = []
        self.auto_sent_ids_at_start = []
        self.max_results = 10
        
        # Initialize JARVIS for email command processing (same as Streamlit version)
        self.jarvis = None
        try:
            self.jarvis = JARVISAssistant()
        except Exception as e:
            print(f"Warning: Could not initialize JARVIS: {e}")
        
        # Load auto-sent IDs
        self._load_auto_sent_ids()
        
        self.setup_ui()
        self.check_environment()
        
    def setup_ui(self):
        """Setup the main user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="📬 Email Agent Desktop", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Left panel for controls
        self.setup_control_panel(main_frame)
        
        # Right panel for email display
        self.setup_email_panel(main_frame)
        
    def setup_control_panel(self, parent):
        """Setup the control panel on the left"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Connection section
        conn_frame = ttk.LabelFrame(control_frame, text="Gmail Connection", padding="5")
        conn_frame.pack(fill="x", pady=(0, 10))
        
        self.connection_label = ttk.Label(conn_frame, text="Not Connected", foreground="red")
        self.connection_label.pack(anchor="w")
        
        self.connect_btn = ttk.Button(conn_frame, text="🔐 Connect to Gmail", 
                                     command=self.connect_gmail)
        self.connect_btn.pack(fill="x", pady=(5, 0))
        
        # Email loading section
        load_frame = ttk.LabelFrame(control_frame, text="Email Loading", padding="5")
        load_frame.pack(fill="x", pady=(0, 10))
        
        # Max results slider
        ttk.Label(load_frame, text="Emails to load:").pack(anchor="w")
        self.max_results_var = tk.IntVar(value=self.max_results)
        self.max_results_scale = ttk.Scale(load_frame, from_=5, to=50, 
                                          variable=self.max_results_var,
                                          orient="horizontal")
        self.max_results_scale.pack(fill="x", pady=(2, 5))
        
        self.max_results_label = ttk.Label(load_frame, text=f"Max: {self.max_results}")
        self.max_results_label.pack(anchor="w")
        self.max_results_scale.configure(command=self.update_max_results_label)
        
        self.load_btn = ttk.Button(load_frame, text="📥 Load Emails", 
                                  command=self.load_emails, state="disabled")
        self.load_btn.pack(fill="x", pady=(5, 0))
        
        # Environment section
        env_frame = ttk.LabelFrame(control_frame, text="Environment", padding="5")
        env_frame.pack(fill="x", pady=(0, 10))
        
        self.groq_status_label = ttk.Label(env_frame, text="GROQ_API_KEY: Not Set", 
                                          foreground="red")
        self.groq_status_label.pack(anchor="w")
        
        # JARVIS Command section (same as Streamlit)
        command_frame = ttk.LabelFrame(control_frame, text="JARVIS Email Command", padding="5")
        command_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(command_frame, text="Send email via command:", font=("Arial", 9)).pack(anchor="w")
        self.command_entry = ttk.Entry(command_frame, width=30)
        self.command_entry.pack(fill="x", pady=(2, 5))
        self.command_entry.insert(0, "email hi to user@example.com")
        
        command_btn = ttk.Button(command_frame, text="📧 Send via JARVIS", 
                                command=self.send_via_jarvis_command)
        command_btn.pack(fill="x", pady=(2, 0))
        
        ttk.Label(command_frame, text="Examples:", font=("Arial", 8), foreground="gray").pack(anchor="w", pady=(5, 0))
        ttk.Label(command_frame, text="• email hi to user@example.com", font=("Arial", 7), foreground="gray").pack(anchor="w")
        ttk.Label(command_frame, text="• send email to first email", font=("Arial", 7), foreground="gray").pack(anchor="w")
        
        # Status section
        status_frame = ttk.LabelFrame(control_frame, text="Status", padding="5")
        status_frame.pack(fill="both", expand=True)
        
        self.status_text = scrolledtext.ScrolledText(status_frame, height=8, width=30)
        self.status_text.pack(fill="both", expand=True)
        
    def setup_email_panel(self, parent):
        """Setup the email display panel on the right"""
        email_frame = ttk.LabelFrame(parent, text="Emails", padding="10")
        email_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create notebook for email display
        self.notebook = ttk.Notebook(email_frame)
        self.notebook.pack(fill="both", expand=True)
        
        # Create initial tab
        self.email_list_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.email_list_frame, text="Email List")
        
        # Email list with scrollbar
        list_frame = ttk.Frame(self.email_list_frame)
        list_frame.pack(fill="both", expand=True)
        
        # Create treeview for email list
        columns = ("From", "Subject", "Classification", "Status")
        self.email_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        self.email_tree.heading("From", text="From")
        self.email_tree.heading("Subject", text="Subject")
        self.email_tree.heading("Classification", text="Classification")
        self.email_tree.heading("Status", text="Status")
        
        self.email_tree.column("From", width=200)
        self.email_tree.column("Subject", width=300)
        self.email_tree.column("Classification", width=120)
        self.email_tree.column("Status", width=150)
        
        # Scrollbar for treeview
        tree_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.email_tree.yview)
        self.email_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.email_tree.pack(side="left", fill="both", expand=True)
        tree_scrollbar.pack(side="right", fill="y")
        
        # Bind selection event
        self.email_tree.bind("<<TreeviewSelect>>", self.on_email_select)
        
        # Email detail frame (initially hidden)
        self.email_detail_frame = ttk.Frame(self.notebook)
        
    def update_max_results_label(self, value):
        """Update the max results label"""
        self.max_results = int(float(value))
        self.max_results_label.config(text=f"Max: {self.max_results}")
        
    def check_environment(self):
        """Check environment variables"""
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            self.groq_status_label.config(text="GROQ_API_KEY: Set", foreground="green")
        else:
            self.groq_status_label.config(text="GROQ_API_KEY: Not Set", foreground="red")
            
    def log_status(self, message):
        """Log status message"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
        
    def connect_gmail(self):
        """Connect to Gmail"""
        def connect_thread():
            try:
                self.log_status("Connecting to Gmail...")
                service = get_service()
                self.connected = True
                self.connection_label.config(text="Connected", foreground="green")
                self.load_btn.config(state="normal")
                self.connect_btn.config(state="disabled")
                self.log_status("Gmail connected successfully!")
            except Exception as e:
                self.log_status(f"Gmail connection failed: {e}")
                messagebox.showerror("Connection Error", f"Failed to connect to Gmail:\n{e}")
                
        threading.Thread(target=connect_thread, daemon=True).start()
        
    def load_emails(self):
        """Load and process emails"""
        if not self.connected:
            messagebox.showwarning("Not Connected", "Please connect to Gmail first.")
            return
            
        def load_thread():
            try:
                self.log_status("Loading emails...")
                
                # Clear previous data
                self.emails = []
                self.results = []
                for item in self.email_tree.get_children():
                    self.email_tree.delete(item)
                    
                # Load emails
                emails_loaded = fetch_recent_emails(get_service(), max_results=self.max_results)
                self.emails = emails_loaded
                
                self.log_status(f"Loaded {len(emails_loaded)} emails")
                
                # Process each email
                for idx, email in enumerate(emails_loaded):
                    try:
                        self.log_status(f"Processing email {idx + 1}/{len(emails_loaded)}...")
                        classification, action = process_email(email)
                        result = {"classification": classification, "action": action}
                        self.results.append(result)
                        
                        # Check if auto-reply should be sent
                        if classification == "normal":
                            self._handle_auto_reply(idx, email, result)
                            
                        # Add to treeview
                        status = self._get_status_text(idx, email, result)
                        self.email_tree.insert("", "end", values=(
                            email.get("from", "")[:50],
                            email.get("subject", "")[:50],
                            classification.upper(),
                            status
                        ))
                        
                    except Exception as err:
                        self.log_status(f"Error processing email {idx + 1}: {err}")
                        error_result = {"classification": "error", "action": f"Processing failed: {err}"}
                        self.results.append(error_result)
                        self.email_tree.insert("", "end", values=(
                            email.get("from", "")[:50],
                            email.get("subject", "")[:50],
                            "ERROR",
                            "Processing failed"
                        ))
                
                self.log_status("Email processing complete!")
                
            except Exception as e:
                self.log_status(f"Failed to load emails: {e}")
                messagebox.showerror("Error", f"Failed to load emails:\n{e}")
                
        threading.Thread(target=load_thread, daemon=True).start()
        
    def _handle_auto_reply(self, idx, email, result):
        """Handle automatic reply for normal emails"""
        try:
            msg_id = email.get("id")
            if msg_id and msg_id in self.auto_sent_ids:
                return  # Already sent
                
            my_email = get_user_email()
            to_addr = self._extract_email_address(email.get("from", ""))
            
            # Skip if sender is the authenticated user
            if my_email and to_addr and to_addr.lower() == my_email.lower():
                result["auto_skipped_self"] = True
                return
                
            original_subject = email.get('subject', '')
            if not original_subject.lower().startswith('re:'):
                subject = f"Re: {original_subject}"
            else:
                subject = original_subject
            body = result.get("action", "") if isinstance(result.get("action"), str) else ""
            
            if body:
                # Use JARVIS's email handling for consistency with Streamlit version
                try:
                    service = get_service()
                    send_email(to_addr, subject, body, service_override=service)
                    if msg_id:
                        self.auto_sent_ids.append(msg_id)
                        self._save_auto_sent_ids()
                    result["auto_sent"] = True
                except Exception as send_err:
                    result["send_error"] = str(send_err)
                
        except Exception as e:
            result["send_error"] = str(e)
            
    def _extract_email_address(self, text: str) -> str:
        """Extract email address from text"""
        try:
            match = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", text or "")
            return match.group(1) if match else (text or "")
        except Exception:
            return text or ""
            
    def _get_status_text(self, idx, email, result):
        """Get status text for email"""
        classification = result.get("classification", "")
        msg_id = email.get("id")
        
        if classification == "normal":
            if msg_id in self.auto_sent_ids_at_start:
                return "Already sent once"
            elif result.get("auto_sent"):
                return "✅ Auto-sent"
            elif result.get("auto_skipped_self"):
                return "↪️ Skipped (self)"
            elif result.get("send_error"):
                return f"Send failed"
            else:
                return "Ready"
        elif classification == "reply_needed":
            return "Reply needed"
        elif classification == "important":
            return "Important"
        elif classification == "spam":
            return "Spam"
        else:
            return "Processed"
            
    def on_email_select(self, event):
        """Handle email selection"""
        selection = self.email_tree.selection()
        if not selection:
            return
            
        # Get selected item
        item = self.email_tree.item(selection[0])
        idx = self.email_tree.index(selection[0])
        
        if idx < len(self.emails):
            self.show_email_detail(idx)
            
    def show_email_detail(self, idx):
        """Show detailed email view"""
        if idx >= len(self.emails) or idx >= len(self.results):
            return
            
        email = self.emails[idx]
        result = self.results[idx]
        
        # Remove existing detail tab if it exists
        for tab_id in self.notebook.tabs():
            if self.notebook.tab(tab_id, "text") == "Email Detail":
                self.notebook.forget(tab_id)
                break
                
        # Create new detail tab
        detail_frame = ttk.Frame(self.notebook)
        self.notebook.add(detail_frame, text="Email Detail")
        self.notebook.select(detail_frame)
        
        # Email details
        details_frame = ttk.LabelFrame(detail_frame, text="Email Details", padding="10")
        details_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(details_frame, text=f"From: {email.get('from', '')}", 
                 font=("Arial", 10, "bold")).pack(anchor="w")
        ttk.Label(details_frame, text=f"Subject: {email.get('subject', '')}", 
                 font=("Arial", 10, "bold")).pack(anchor="w")
        ttk.Label(details_frame, text="Body:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 0))
        
        body_text = scrolledtext.ScrolledText(details_frame, height=6, width=80)
        body_text.pack(fill="x", pady=(5, 0))
        body_text.insert("1.0", email.get("body", ""))
        body_text.config(state="disabled")
        
        # Classification and action
        classification = result.get("classification", "")
        action = result.get("action", "")
        
        class_frame = ttk.LabelFrame(detail_frame, text="Classification & Action", padding="10")
        class_frame.pack(fill="x", padx=10, pady=10)
        
        classification_color = self._get_classification_color(classification)
        class_label = ttk.Label(class_frame, text=f"Classification: {classification.upper()}", 
                               font=("Arial", 10, "bold"), foreground=classification_color)
        class_label.pack(anchor="w")
        
        if action:
            ttk.Label(class_frame, text="Suggested Action:", 
                     font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 0))
            action_text = scrolledtext.ScrolledText(class_frame, height=8, width=80)
            action_text.pack(fill="x", pady=(5, 0))
            action_text.insert("1.0", action)
            action_text.config(state="disabled")
        
        # Reply section (for emails that need replies)
        if classification in ["reply_needed", "normal", "important"]:
            reply_frame = ttk.LabelFrame(detail_frame, text="Send Reply", padding="10")
            reply_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Reply form
            ttk.Label(reply_frame, text="To:").grid(row=0, column=0, sticky="w", pady=(0, 5))
            to_entry = ttk.Entry(reply_frame, width=60)
            to_entry.grid(row=0, column=1, sticky="ew", pady=(0, 5))
            # Extract email address from "From" field (same logic as Streamlit/JARVIS)
            from_field = email.get("from", "")
            email_addr = self._extract_email_address(from_field)
            to_entry.insert(0, email_addr if email_addr else from_field)
            
            ttk.Label(reply_frame, text="Subject:").grid(row=1, column=0, sticky="w", pady=(0, 5))
            subject_entry = ttk.Entry(reply_frame, width=60)
            subject_entry.grid(row=1, column=1, sticky="ew", pady=(0, 5))
            # Handle "Re:" prefix properly (same as JARVIS logic)
            original_subject = email.get('subject', '')
            if not original_subject.lower().startswith('re:'):
                reply_subject = f"Re: {original_subject}"
            else:
                reply_subject = original_subject
            subject_entry.insert(0, reply_subject)
            
            ttk.Label(reply_frame, text="Body:").grid(row=2, column=0, sticky="nw", pady=(0, 5))
            body_entry = scrolledtext.ScrolledText(reply_frame, height=10, width=60)
            body_entry.grid(row=2, column=1, sticky="nsew", pady=(0, 5))
            body_entry.insert("1.0", action if isinstance(action, str) else "")
            
            # Send button
            send_btn = ttk.Button(reply_frame, text="✉️ Send Reply", 
                                 command=lambda: self.send_reply(to_entry.get(), 
                                                               subject_entry.get(), 
                                                               body_entry.get("1.0", tk.END)))
            send_btn.grid(row=3, column=1, sticky="e", pady=(10, 0))
            
            # Configure grid weights
            reply_frame.columnconfigure(1, weight=1)
            reply_frame.rowconfigure(2, weight=1)
            
    def _get_classification_color(self, classification):
        """Get color for classification"""
        colors = {
            "spam": "red",
            "important": "blue", 
            "normal": "green",
            "reply_needed": "orange",
            "error": "red"
        }
        return colors.get(classification, "black")
        
    def process_email_command(self, command_text: str) -> str:
        """Process email command through JARVIS (same as Streamlit version)"""
        try:
            if not self.jarvis:
                return "❌ JARVIS not initialized. Cannot process email command."
            
            # Create command object (same as Streamlit's process_chat_message)
            command = Command(
                type=CommandType.TEXT,
                content=command_text.strip(),
                source="desktop_app",
                timestamp=datetime.now()
            )
            
            # Process through JARVIS (same as Streamlit's process_chat_message)
            response = self.jarvis.process_command(command)
            
            # Learn from this interaction (same as Streamlit)
            self.jarvis.learn_from_interaction(command, response)
            
            return response
            
        except Exception as e:
            return f"❌ Error processing email command: {e}"
    
    def send_via_jarvis_command(self):
        """Send email via JARVIS command (same as Streamlit version)"""
        command_text = self.command_entry.get().strip()
        if not command_text:
            messagebox.showwarning("Empty Command", "Please enter an email command.")
            return
        
        def send_thread():
            try:
                self.log_status(f"Processing JARVIS command: {command_text}")
                response = self.process_email_command(command_text)
                self.log_status(f"JARVIS Response: {response}")
                
                # Show result in messagebox
                if "sent" in response.lower() and "error" not in response.lower() and "failed" not in response.lower():
                    messagebox.showinfo("Success", f"Email command executed successfully!\n\n{response}")
                else:
                    messagebox.showinfo("Result", f"Command Result:\n\n{response}")
                    
            except Exception as e:
                self.log_status(f"Error processing command: {e}")
                messagebox.showerror("Error", f"Failed to process command:\n{e}")
        
        threading.Thread(target=send_thread, daemon=True).start()
    
    def send_reply(self, to_addr, subject, body):
        """Send reply email using JARVIS command processing (same as Streamlit version)"""
        if not to_addr or not subject or not body.strip():
            messagebox.showwarning("Missing Information", 
                                 "Please fill in all fields (To, Subject, Body)")
            return
        
        # Extract email address from "Name <email@example.com>" format (same as Streamlit/JARVIS)
        to_addr_clean = self._extract_email_address(to_addr)
        if not to_addr_clean:
            messagebox.showerror("Invalid Email", f"Could not extract email address from: {to_addr}")
            return
            
        def send_thread():
            try:
                self.log_status(f"Sending reply to {to_addr_clean}...")
                
                # Use JARVIS command processing (same as Streamlit version)
                # This uses the same handle_gmail_command logic that works in Streamlit
                if self.jarvis:
                    # Format command like Streamlit does: "email <message> to <address>"
                    command_text = f"email {body.strip()} to {to_addr_clean}"
                    response = self.process_email_command(command_text)
                    
                    # Check if email was sent successfully
                    if "sent" in response.lower() and "error" not in response.lower() and "failed" not in response.lower():
                        self.log_status(f"Reply sent successfully via JARVIS!")
                        messagebox.showinfo("Success", f"Reply sent successfully!\n\n{response}")
                    else:
                        # If JARVIS command didn't work, try direct send as fallback
                        self.log_status("JARVIS command didn't match pattern, using direct send...")
                        service = get_service()
                        response = send_email(to_addr_clean, subject, body.strip(), service_override=service)
                        if response and response.get('id'):
                            self.log_status(f"Reply sent successfully! Message ID: {response.get('id')}")
                            messagebox.showinfo("Success", f"Reply sent successfully!\n\nTo: {to_addr_clean}\nSubject: {subject}\nMessage ID: {response.get('id')}")
                        else:
                            messagebox.showinfo("Success", "Reply sent successfully!")
                else:
                    # Fallback: direct send if JARVIS not available
                    service = get_service()
                    response = send_email(to_addr_clean, subject, body.strip(), service_override=service)
                    if response and response.get('id'):
                        self.log_status(f"Reply sent successfully! Message ID: {response.get('id')}")
                        messagebox.showinfo("Success", f"Reply sent successfully!\n\nTo: {to_addr_clean}\nSubject: {subject}\nMessage ID: {response.get('id')}")
                    else:
                        messagebox.showinfo("Success", "Reply sent successfully!")
                    
            except Exception as e:
                self.log_status(f"Failed to send reply: {e}")
                messagebox.showerror("Send Error", f"Failed to send reply:\n{e}")
                
        threading.Thread(target=send_thread, daemon=True).start()
        
    def _load_auto_sent_ids(self):
        """Load auto-sent email IDs"""
        try:
            if os.path.exists("auto_sent.json"):
                with open("auto_sent.json", "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.auto_sent_ids = data
                        self.auto_sent_ids_at_start = list(data)
        except Exception:
            self.auto_sent_ids = []
            self.auto_sent_ids_at_start = []
            
    def _save_auto_sent_ids(self):
        """Save auto-sent email IDs"""
        try:
            with open("auto_sent.json", "w") as f:
                json.dump(list(dict.fromkeys(self.auto_sent_ids)), f, indent=2)
        except Exception:
            pass


def main():
    root = tk.Tk()
    app = EmailAgentDesktop(root)
    root.mainloop()


if __name__ == "__main__":
    main()
