#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as msg
from tkcalendar import *
import datetime
import pandas as pd
from helper import *



model = Model()
model_list = model.get_classifier_types()

PADX = 10
PADY = 10

window = tk.Tk()
window.title("Fraud detection demo")
window.geometry('500x500')
window.resizable(0,0)

tk.Label(window, text='Transaction Date :').grid(column=0, row=0, sticky=tk.E, padx=PADX, pady=PADY)
tk.Label(window, text='Transaction Time :').grid(column=0, row=1, sticky=tk.E, padx=PADX, pady=PADY)
tk.Label(window, text='Transaction Amount :').grid(column=0, row=2, sticky=tk.E, padx=PADX, pady=PADY)
tk.Label(window, text='Customer ID :').grid(column=0, row=3, sticky=tk.E, padx=PADX, pady=PADY)
tk.Label(window, text='Terminal ID :').grid(column=0, row=4, sticky=tk.E, padx=PADX, pady=PADY)

err2 = tk.Label(window, text='*', fg='red', font=20)
err2.grid_forget()

cal = Calendar(window, selectmode="day", year=2018, month=7, day=26, showweeknumbers=False,
               showothermonthdays=False, date_pattern='y-mm-dd')
cal.grid(column=1, row=0, sticky=tk.W, padx=PADX, pady=PADY)

time_frame = tk.Frame(window)
hour = tk.Spinbox(time_frame, from_=0, to=23, width=2, state="readonly", justify=tk.CENTER, format="%02.0f")
minute = tk.Spinbox(time_frame, from_=0, to=59, width=2, state="readonly", justify=tk.CENTER, format="%02.0f")
second = tk.Spinbox(time_frame, from_=0, to=59, width=2, state="readonly", justify=tk.CENTER, format="%02.0f")
hour.grid(column=0, row=0, sticky=tk.W, ipadx=PADX, ipady=PADY)
minute.grid(column=1, row=0, sticky=tk.W, ipadx=PADX, ipady=PADY)
second.grid(column=2, row=0, sticky=tk.W, ipadx=PADX, ipady=PADY)
time_frame.grid(column=1, row=1, sticky=tk.W, padx=PADX, pady=PADY)

entry2 = tk.Entry(window, text='entry2')
entry2.grid(column=1, row=2, sticky=tk.W, padx=PADX, pady=PADY)
entry2.insert(0,'0.0')

customer_ids = model.get_customer_ids()
customer_id_cb = ttk.Combobox(window, values=customer_ids, state='readonly')
customer_id_cb.current(0)
customer_id_cb.grid(column=1, row=3, sticky=tk.W, padx=PADX, pady=PADY)

terminal_ids = model.get_terminal_ids()
terminal_id_cb = ttk.Combobox(window, values=terminal_ids, state='readonly')
terminal_id_cb.current(0)
terminal_id_cb.grid(column=1, row=4, sticky=tk.W, padx=PADX, pady=PADY)

radioVar = tk.IntVar()
radioVar.set(0)

radio_frame = tk.Frame(window)
radio_options = [(model_list[0], 0), (model_list[1], 1), (model_list[2], 2), (model_list[3], 3)]
for num,(text, value) in enumerate(radio_options):
    tk.Radiobutton(radio_frame,
                  text=text,
                  variable=radioVar,
                  value=value,
                  indicator=0,
                  padx=5, pady=5).grid(column=num,row=0, sticky=tk.NS, padx=PADX, pady=PADY)

radio_frame.grid(column=0, row=5, columnspan=3, padx=PADX, pady=PADY)

def is_amount_valid(amount):
    try:
        float(amount)
        return True
    except ValueError:
        return False
    
def buttonCmd():
    
    errMessage = ''
    infoMessage = ''
    showErr = False
    txt_date_time = cal.get_date()+" "+hour.get() + ':' + minute.get() + ':' + second.get()
    txt_amount = entry2.get()
    txt_customerID = customer_id_cb.get() 
    txt_terminalID = terminal_id_cb.get()    
    
    
    # validation of amount
    if is_amount_valid(txt_amount):
        txt_amount = float(txt_amount)
        err2.grid_forget()
    else:
        err2.grid(column=3, row=1)
        showErr = True
        errMessage += 'invalid Amount. (FLOAT)\n'
        

    # show messages
    if showErr:
        showErr = False
        msg.showerror('Invalid Input!', errMessage)
    else:
        transaction_data = []
        transaction_data.append([
            txt_date_time ,
            txt_customerID ,
            txt_terminalID ,
            txt_amount])
        transaction_data = pd.DataFrame(transaction_data, columns=['TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT'])
        transaction_data['TX_DATETIME'] = transaction_data['TX_DATETIME'].astype('datetime64')
        infoMessage = "The transaction is {:.2f} % fraud.".format(float(model.predict(model_list[radioVar.get()], transaction_data))*100)
        msg.showinfo('Prediction Result', infoMessage)

        
    
tk.Button(window, text='Predict', command=buttonCmd, padx=5, pady=5).grid(column=0, row=6, columnspan=3, sticky=tk.N, padx=PADX, pady=PADY)



window.mainloop()

