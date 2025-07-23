import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --------- PAGE CONFIG ---------
st.set_page_config(page_title="AI-Powered Energy Calculator & Forecast", page_icon="⚡", layout="centered")

# --------- CUSTOM STYLE ---------
st.markdown("""
    <style>
    body {
        background-color: #e6f0ff;
    }
    .stApp {
        background-color: #e6f0ff;
    }
    .stButton button {
        background-color: #e6f0ff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
            background-color: #e6f0ff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    .stRadio > div {
        flex-direction: row;
    }
    .block-container {
        padding-top: 2rem;
    }
     .stSelectbox > div, .stDateInput > div {
        background: white !important;
        border-radius: 10px;
        padding: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --------- SESSION STATE ---------
if "total_appliance_kwh" not in st.session_state:
    st.session_state.total_appliance_kwh = 0

# --------- HEADER ---------
st.title("⚡ Smart Energy Forecast & Appliance Calculator")
st.caption("Plan smarter, save better. " \
"Forecast your energy usage and optimize appliance consumption.")

# --------- SIDEBAR ---------
with st.sidebar:
    st.header("⚙️ Configuration")
    mode = st.radio("Choose mode:", ["Calculator Only", "Forecast with Daily Usage"])
    duration = st.selectbox("Forecast/Calculation Period:", ["1 Day", "7 Days", "30 Days"])
    days_to_predict = int(duration.split()[0])
    start_date = st.date_input("📅 Forecast Start Date")

# --------- USER ENERGY INPUT ---------
with st.container():
    st.subheader("🔍 Your Actual Energy Usage")
    energy_period = st.selectbox("Time period of your energy usage:", ["Daily", "Weekly", "Monthly"])
    user_total_energy = st.number_input(f"Enter your actual energy usage ({energy_period}) in kWh", min_value=0.0)

# --------- APPLIANCE INPUT ---------
with st.container():
    st.subheader("🔌 Appliance Input")
    num_appliances = st.slider("How many appliances do you want to enter?", min_value=1, max_value=20, value=3)
    custom_appliances = []

    with st.form("appliance_form"):
        for i in range(num_appliances):
            st.markdown(f"**Appliance {i + 1}**")
            name = st.text_input("Name", key=f"name_{i}")
            power = st.number_input("Power (Watts)", min_value=0, key=f"power_{i}")
            hours = st.number_input("Hours used per day", min_value=0.0, max_value=24.0, step=0.5, key=f"hours_{i}")
            custom_appliances.append((name, power, hours))
            st.divider()
        calculate_appliances = st.form_submit_button("🔢 Calculate Appliance Usage")

# --------- CALCULATE USAGE ---------
if calculate_appliances:
    st.session_state.total_appliance_kwh = 0
    st.subheader("📊 Appliance Consumption Breakdown")
    for name, power, hours in custom_appliances:
        energy = round((power * hours * days_to_predict) / 1000, 2)
        st.session_state.total_appliance_kwh += energy
        st.write(f"- **{name}**: {energy:.2f} kWh over {days_to_predict} day(s)")
    st.success(f"✅ Total Appliance Usage: **{st.session_state.total_appliance_kwh:.2f} kWh**")

    pie_labels = [ap[0] for ap in custom_appliances]
    pie_values = [(ap[1] * ap[2] * days_to_predict) / 1000 for ap in custom_appliances]
    fig1, ax1 = plt.subplots()
    ax1.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

# --------- COST ESTIMATOR ---------
with st.container():
    st.subheader("💰 Estimate Energy Cost")
    with st.form("cost_form"):
        rate = st.number_input("Cost per kWh (₦)", min_value=0.0, step=0.01, format="%.2f")
        estimate_btn = st.form_submit_button("Estimate Cost")
    if estimate_btn:
        total_kwh = st.session_state.total_appliance_kwh
        if total_kwh == 0 or rate == 0.0:
            st.warning("⚠️ Ensure appliance usage and rate are provided.")
        else:
            cost = total_kwh * rate
            st.success(f"🔌 Estimated Energy Cost: ₦{cost:,.2f}")

# --------- CONTRIBUTION ---------
if user_total_energy > 0 and st.session_state.total_appliance_kwh > 0:
    st.subheader("📈 Appliance Contribution to Total Usage")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Actual Usage", f"{user_total_energy:.2f} kWh")
    with col2:
        st.metric("Appliances", f"{st.session_state.total_appliance_kwh:.2f} kWh")
    for name, power, hours in custom_appliances:
        app_kwh = round((power * hours * days_to_predict) / 1000, 2)
        percent = (app_kwh / user_total_energy) * 100
        st.write(f"- **{name}** → {app_kwh:.2f} kWh ({percent:.1f}%)")
    if st.session_state.total_appliance_kwh > user_total_energy:
        st.subheader("💡 Suggested Reductions")
        appliance_savings = sorted(
            [(name, power, hours, round((power * hours * days_to_predict) / 1000, 2)) 
             for name, power, hours in custom_appliances],
            key=lambda x: x[3], reverse=True
        )
        for name, power, hours, energy_kwh in appliance_savings[:3]:
            if hours >= 2:
                reduced_hours = hours - 1
                reduced_kwh = round((power * reduced_hours * days_to_predict) / 1000, 2)
                saved = round(energy_kwh - reduced_kwh, 2)
                st.write(f"🔻 **{name}** → Reduce to {reduced_hours:.1f}h/day → Save **{saved:.2f} kWh**")
    else:
        st.success("✅ Usage within expected range")

# --------- FORECAST ---------
if mode == "Forecast with Daily Usage":
    st.subheader("🔮 Forecast Your Daily Usage")
    daily_usage = [st.number_input(f"Day {i} Usage (kWh)", min_value=0.0, key=f"day_{i}") for i in range(1, days_to_predict + 1)]
    if st.button("📊 Run Forecast"):
        if len(set(daily_usage)) <= 1:
            st.warning("⚠️ Enter at least two unique daily values.")
        else:
            df = pd.DataFrame({"Day": list(range(1, days_to_predict + 1)), "Energy_Usage_kWh": daily_usage})
            df["Next_Day_Usage"] = df["Energy_Usage_kWh"].shift(-1)
            df.dropna(inplace=True)
            model = LinearRegression()
            model.fit(df[["Energy_Usage_kWh"]], df["Next_Day_Usage"])

            predictions = []
            last_val = daily_usage[-1]
            for _ in range(days_to_predict):
                pred = model.predict([[last_val]])[0]
                predictions.append(round(pred, 2))
                last_val = pred

            st.subheader("📈 Forecasted Usage")
            st.write(pd.DataFrame({"Day": range(1, days_to_predict + 1), "Forecast": predictions}))
            fig, ax = plt.subplots()
            ax.plot(range(1, days_to_predict + 1), predictions, marker='o', linestyle='--', color='orange')
            ax.set_title(f"{days_to_predict}-Day Forecast")
            ax.set_xlabel("Day")
            ax.set_ylabel("Predicted kWh")
            ax.grid(True)
            st.pyplot(fig)

            total_forecast = sum(predictions)
            st.subheader("🔍 Forecast vs Appliance Usage")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Forecasted", f"{total_forecast:.2f} kWh")
            with col2:
                st.metric("Appliances", f"{st.session_state.total_appliance_kwh:.2f} kWh")

                fig2, ax2 = plt.subplots()
            ax2.bar(["Forecasted", "Appliances"], [total_forecast, st.session_state.total_appliance_kwh], color=['orange', 'skyblue'])
            ax2.set_ylabel("kWh")
            st.pyplot(fig2)

            if st.session_state.total_appliance_kwh > total_forecast:
                with st.expander("💡 Suggested Reductions"):
                    appliance_savings = sorted(
                        [(name, power, hours, round((power * hours * days_to_predict) / 1000, 2))
                         for name, power, hours in custom_appliances],
                        key=lambda x: x[3], reverse=True
                    )
                    for name, power, hours, kwh in appliance_savings[:3]:
                        if hours >= 2:
                            reduced_hours = hours - 1
                            reduced_kwh = round((power * reduced_hours * days_to_predict) / 1000, 2)
                            saved = round(kwh - reduced_kwh, 2)
                            st.write(f"🔻 Reduce **{name}** to {reduced_hours:.1f}h/day → Save **{saved:.2f} kWh**")
            else:
                st.success("✅ You're within the forecasted usage!")


