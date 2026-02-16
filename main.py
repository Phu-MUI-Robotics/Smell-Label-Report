# 1.Import
from datetime import datetime, time, timedelta
from influxdb import InfluxDBClient
from dotenv import load_dotenv
import os
import io
import math
import streamlit as st
import pytz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from labeling import radar_peak_window, pca_consistency_plot, box_plot, create_report_package

# Load env.
load_dotenv()

# 2.InfluxDB
# Confiig
host = os.getenv("INFLUXDB_HOST") or "localhost"
port_str = os.getenv("INFLUXDB_PORT")
port = int(port_str) if port_str is not None else 8086
username = os.getenv("INFLUXDB_USER") or ""
password = os.getenv("INFLUXDB_PASS") or ""
database = os.getenv("INFLUXDB_DB") or ""

client = InfluxDBClient(host=host, port=port, username=username, password=password, database=database)

# Connect
def connect_influxdb():
    try:
        client = InfluxDBClient(host=host, port=port, username=username, password=password, database=database)
        client.get_list_database()
        print("[DEBUG] Connected to InfluxDB successfully.")
        return client
    except Exception as e:
        print(f"[ERROR] Failed to connect to InfluxDB: {e}")
        return None

def get_measurements(client):
    try:
        result = client.query('SHOW MEASUREMENTS')
        measurements = [m['name'] for m in result.get_points()]
        return measurements
    except Exception as e:
        print(f"[ERROR] Failed to query measurements: {e}")
        return []

def get_serial_numbers(client, measurement):
    try:
        query = f'SHOW TAG VALUES FROM "{measurement}" WITH KEY = "sn"'
        result = client.query(query)
        serials = [point['value'] for point in result.get_points()]
        return serials
    except Exception as e:
        print(f"[ERROR] Failed to query serial numbers: {e}")
        return []
    
def get_station_names(client, measurement):
    try:
        query = f'SHOW TAG VALUES FROM "{measurement}" WITH KEY = "sName"'
        result = client.query(query)
        stations = [point['value'] for point in result.get_points()]
        return stations
    except Exception as e:
        print(f"[ERROR] Failed to query station names: {e}")
        return []

# 3.Streamlit App

client = connect_influxdb()
if client:
    measurements = get_measurements(client)
else:
    measurements = []
    
st.title("Smell Label Mini-App")
st.markdown("")
st.markdown("")
st.subheader("1. เลือก Measurement -> Serial No./Station Name")
st.markdown("")

# 1) เลือก Measurement -> Serial No./Station Name
def serial_sort_key(sn):
    try:
        parts = sn.split('-')
        if len(parts) >= 3:
            month = int(parts[1][:2])
            year = int(parts[1][2:])
            return (year, month, sn)
    except Exception:
        pass
    return (0, 0, sn)

if not measurements:
    measurements = []
    selected_measurement = st.selectbox("กรุณาเลือก Measurement :", measurements, index=0)
    serial_numbers = []
    unique_serial_numbers = ["-"]
    selected_sn = st.selectbox("กรุณาเลือก Serial No. :", unique_serial_numbers, disabled=True)
    selected_station = None
else:
    measurements = ["-"] + measurements
    selected_measurement = st.selectbox("กรุณาเลือก Measurement :", measurements, index=0)
    serial_numbers = []
    if client and selected_measurement != "-":
        serial_numbers = get_serial_numbers(client, selected_measurement)
    
    unique_serial_numbers = sorted(set(serial_numbers), key=serial_sort_key) if serial_numbers else []
    if unique_serial_numbers:
        unique_serial_numbers = ["-"] + unique_serial_numbers + ["❌ ไม่เจอ - ค้นหาจาก Station"]
    else:
        unique_serial_numbers = ["-", "❌ ไม่เจอ - ค้นหาจาก Station"]
    selected_sn = st.selectbox("กรุณาเลือก Serial No. :", unique_serial_numbers)
    
    # ถ้าเลือก "ไม่เจอ" ให้แสดง dropdown Station
    selected_station = None
    if selected_sn == "❌ ไม่เจอ - ค้นหาจาก Station":
        if client and selected_measurement != "-":
            station_names = get_station_names(client, selected_measurement)
            unique_stations = sorted(set(station_names)) if station_names else ["-"]
            if unique_stations and unique_stations != "-":
                unique_stations = ["-"] + unique_stations   
            selected_station = st.selectbox("🔍 กรุณาเลือก Station :", unique_stations)
            if selected_station != "-":
                selected_sn = selected_station
        else:
            st.warning("⚠️ กรุณาเลือก Measurement ก่อน")
            selected_sn = "-"
st.write(f"Measurement ที่เลือก : {selected_measurement}")
if selected_station:
    st.write(f"🔍 ค้นหาจาก Station : {selected_station}")
else:
    st.write(f"Serial No. ที่เลือก : {selected_sn}")

# 2) เลือกช่วงเวลา
st.markdown("")
st.subheader("2. เลือกช่วงเวลา")
st.markdown("")

time_precision = st.selectbox("เลือกความละเอียดของเวลา : ", ["นาที (00:00)", "วินาที (00:00:00)"], index=0)
is_second = time_precision == "วินาที (00:00:00)"
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("วันที่เริ่มต้น :", value = datetime.now().date()-timedelta(days=1))
    if is_second:
        colh, colm, cols = st.columns(3)
        hour_options = [str(h).zfill(2) for h in range(0,24)]
        minute_options = [str(m).zfill(2) for m in range(0,60)]
        second_options = [str(s).zfill(2) for s in range(0,60)]
        start_hour_str = colh.selectbox("ชั่วโมงเริ่มต้น", hour_options, index=0, key="start_hour")
        start_minute_str = colm.selectbox("นาทีเริ่มต้น", minute_options, index=0, key="start_minute")
        start_second_str = cols.selectbox("วินาทีเริ่มต้น", second_options, index=0, key="start_second")
        start_time = time(int(start_hour_str), int(start_minute_str), int(start_second_str))
    else:
        start_time = st.time_input("เวลาเริ่มต้น", value=time(0, 0))
with col2:
    end_date = st.date_input("วันที่สิ้นสุด", value=datetime.now().date())
    if is_second:
        colh, colm, cols = st.columns(3)
        hour_options = [str(h).zfill(2) for h in range(0,24)]
        minute_options = [str(m).zfill(2) for m in range(0,60)]
        second_options = [str(s).zfill(2) for s in range(0,60)]
        end_hour_str = colh.selectbox("ชั่วโมงสิ้นสุด", hour_options, index=23, key="end_hour")
        end_minute_str = colm.selectbox("นาทีสิ้นสุด", minute_options, index=59, key="end_minute")
        end_second_str = cols.selectbox("วินาทีสิ้นสุด", second_options, index=59, key="end_second")
        end_time = time(int(end_hour_str), int(end_minute_str), int(end_second_str))
    else:
        end_time = st.time_input("เวลาสิ้นสุด", value=time(23, 59))

#Asia/Bangkok Timezone
# เช็คว่า date ไม่เป็น None
if start_date is None or end_date is None:
    st.error("กรุณาเลือกวันที่เริ่มต้นและวันที่สิ้นสุด")
    st.stop()

bangkok_tz = pytz.timezone('Asia/Bangkok')
start_dt = bangkok_tz.localize(datetime.combine(start_date, start_time))
end_dt = bangkok_tz.localize(datetime.combine(end_date, end_time))
start_unix = int(start_dt.timestamp())
end_unix = int(end_dt.timestamp())


st.write(f"Unix เริ่มต้น : {start_unix}")
st.write(f"Unix สิ้นสุด : {end_unix}")
st.markdown("")

# Add : ชื่อสาร
st.subheader("3. ชื่อสารเคมี")
st.text_input("ชื่อสาร :", key="chemical_name")
st.markdown("")

# 3) Implementation
implement_btn = st.button("Implement", type="primary")

if implement_btn:
    # Prepare query parameters
    tag_key = None
    tag_value = None
    if selected_sn and selected_sn != "-" and selected_sn != "❌ ไม่เจอ - ค้นหาจาก Station":
        tag_key = "sn"
        tag_value = selected_sn
    elif selected_station and selected_station != "-":
        tag_key = "sName"
        tag_value = selected_station
    else:
        st.warning("กรุณาเลือก Serial No. หรือ Station Name ให้ถูกต้อง")
        tag_key = None

    if tag_key and tag_value and selected_measurement and selected_measurement != "-":
        # ใช้ unix timestamp (UTC) สำหรับ query
        start_unix = int(start_dt.timestamp())
        end_unix = int(end_dt.timestamp())
        # Proper regex for sn/sName (no single quotes inside regex)
        if tag_key == "sn":
            tag_filter = f'("sn" =~ /^{tag_value}$/)'
        else:
            tag_filter = f'("sName" =~ /^{tag_value}$/)'
        query = (
            f'SELECT mean("a1") AS "s1", mean("a2") AS "s2", mean("a3") AS "s3", mean("a4") AS "s4", '
            f'mean("a5") AS "s5", mean("a6") AS "s6", mean("a7") AS "s7", mean("a8") AS "s8", mean("total") AS "Total" '
            f'FROM "{selected_measurement}" '
            f'WHERE {tag_filter} AND time >= {start_unix}000ms AND time <= {end_unix}000ms '
            f'GROUP BY time(15s)'
        )
        try:
            result = client.query(query)
            points = list(result.get_points())
            if points:
                df = pd.DataFrame(points)
                # แปลง column time เป็น Asia/Bangkok
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('Asia/Bangkok').dt.strftime('%d/%m/%Y %H:%M:%S')
                    df.rename(columns={'time': 'Time'}, inplace=True)
                
                csv = df.to_csv(index=False)
                
                # Store data in session state
                st.session_state['temp_csv'] = csv
                st.session_state['temp_df'] = df
                st.session_state['data_ready'] = True
                st.success("✅ Query สำเร็จ! พร้อมยืนยันข้อมูล")
            else:
                st.info("ไม่พบข้อมูลในช่วงเวลาที่เลือก")
                st.session_state['data_ready'] = False
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")
            st.session_state['data_ready'] = False
    elif tag_key is None:
        pass  # Already showed warning above
    else:
        st.warning("กรุณาเลือก Measurement, Serial No. หรือ Station Name ให้ถูกต้อง")

# แสดง DataFrame และปุ่ม Confirm (ถ้ามีข้อมูล)
if st.session_state.get('data_ready', False):
    st.markdown("---")
    st.subheader("📋 ข้อมูลที่ Query ได้")
    df_display = st.session_state.get('temp_df')
    if df_display is not None:
        st.dataframe(df_display)
        st.markdown("")
        
        confirm_btn = st.button("Confirm", type="primary")
        if confirm_btn:
            # Save CSV to file and session state
            with open('all_data.csv', 'w', encoding='utf-8') as f:
                f.write(st.session_state['temp_csv'])
            st.session_state['all_data.csv'] = st.session_state['temp_csv']
            st.session_state['show_labeling'] = True
            st.success("✅ Confirmed! เริ่มประมวลผล...")
            st.rerun()

# 4) Labeling & Visualization (แสดงหลังจาก Confirm)
if st.session_state.get('show_labeling', False):
    st.markdown("---")
    st.subheader("📊 Labeling & Visualization")
    
    # Initialize config in session state if not exists
    if 'config' not in st.session_state:
        st.session_state['config'] = {
            'GAUSS_SIGMA': 2.0,
            'MIN_PEAK_DISTANCE': 10,
            'MIN_PROMINENCE': 3,
            'W_LEFT': 3,
            'W_RIGHT': 3
        }
    
    # Config
    TOTAL_COL = "Total"
    TIME_COL = "Time"
    SENSOR_COLS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
    
    # Get config from session state
    GAUSS_SIGMA = st.session_state['config']['GAUSS_SIGMA']
    MIN_PEAK_DISTANCE = st.session_state['config']['MIN_PEAK_DISTANCE']
    MIN_PROMINENCE = st.session_state['config']['MIN_PROMINENCE']
    W_LEFT = st.session_state['config']['W_LEFT']
    W_RIGHT = st.session_state['config']['W_RIGHT']
    
    # Show tuning interface if in tuning mode
    if st.session_state.get('tuning_mode', False):
        st.markdown("### 🎛️ ปรับค่า Config Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            new_gauss = st.number_input("GAUSS_SIGMA", min_value=0.1, max_value=10.0, value=GAUSS_SIGMA, step=0.1)
            new_peak_dist = st.number_input("MIN_PEAK_DISTANCE", min_value=1, max_value=100, value=MIN_PEAK_DISTANCE, step=1)
            new_prominence = st.number_input("MIN_PROMINENCE", min_value=0.1, max_value=50.0, value=float(MIN_PROMINENCE), step=0.5)
        
        with col2:
            new_w_left = st.number_input("W_LEFT", min_value=0, max_value=20, value=W_LEFT, step=1)
            new_w_right = st.number_input("W_RIGHT", min_value=0, max_value=20, value=W_RIGHT, step=1)
        
        st.markdown("")
        apply_btn = st.button("✅ Apply & Visualize", type="primary")
        
        if apply_btn:
            # Update config
            st.session_state['config']['GAUSS_SIGMA'] = new_gauss
            st.session_state['config']['MIN_PEAK_DISTANCE'] = new_peak_dist
            st.session_state['config']['MIN_PROMINENCE'] = new_prominence
            st.session_state['config']['W_LEFT'] = new_w_left
            st.session_state['config']['W_RIGHT'] = new_w_right
            st.session_state['tuning_mode'] = False  # Exit tuning mode
            st.success("✅ Config updated! Visualizing...")
            st.rerun()
        
        st.markdown("---")
    
    # Show current config
    with st.expander("⚙️ Current Config"):
        st.write(f"GAUSS_SIGMA = {GAUSS_SIGMA}")
        st.write(f"MIN_PEAK_DISTANCE = {MIN_PEAK_DISTANCE}")
        st.write(f"MIN_PROMINENCE = {MIN_PROMINENCE}")
        st.write(f"W_LEFT = {W_LEFT}")
        st.write(f"W_RIGHT = {W_RIGHT}")
    
    try:
        # Load data from memory
        df_label = pd.read_csv(io.StringIO(st.session_state['all_data.csv']))
        
        # Check columns
        original_cols = [TOTAL_COL, *SENSOR_COLS]
        missing = [col for col in original_cols if col not in df_label.columns]
        if missing:
            st.error(f"Missing columns in CSV: {missing}")
        else:
            # Convert to numeric
            for c in original_cols:
                df_label[c] = pd.to_numeric(df_label[c], errors='coerce')
            
            total_raw = df_label[TOTAL_COL].to_numpy(dtype=float)
            amount_total_count = len(total_raw)
            
            # Prepare x-axis
            use_time = TIME_COL in df_label.columns
            if use_time:
                x = pd.to_datetime(df_label[TIME_COL], errors="coerce")
                if x.isna().all():
                    use_time = False
            if not use_time:
                x = pd.Series(np.arange(amount_total_count), index=df_label.index)
            
            # Gaussian smoothing + peak detection
            total_gauss = gaussian_filter1d(total_raw, sigma=GAUSS_SIGMA)
            peaks, _ = find_peaks(total_gauss, distance=MIN_PEAK_DISTANCE, prominence=MIN_PROMINENCE)
            
            if len(peaks) == 0:
                st.warning("No peaks detected. Try adjusting MIN_PROMINENCE or MIN_PEAK_DISTANCE.")
            else:
                st.info(f"Detected {len(peaks)} peaks")
                
                # Valley-to-valley cycles
                bounds = np.zeros(len(peaks) + 1, dtype=int)
                bounds[0] = 0
                bounds[-1] = amount_total_count
                
                for i in range(1, len(peaks)):
                    a, b = peaks[i - 1], peaks[i]
                    seg = total_gauss[a:b]
                    bounds[i] = a if len(seg) == 0 else (a + int(np.argmin(seg)))
                
                bounds = np.clip(bounds, 0, amount_total_count)
                bounds = np.maximum.accumulate(bounds)
                
                cycle_id = np.zeros(amount_total_count, dtype=int)
                for i in range(len(peaks)):
                    cycle_id[bounds[i]:bounds[i + 1]] = i + 1
                
                df_label["cycle_id"] = cycle_id
                
                # Peak window extraction
                keep_mask = np.zeros(amount_total_count, dtype=bool)
                
                for cid in range(1, len(peaks) + 1):
                    sb, eb = bounds[cid - 1], bounds[cid]
                    p = peaks[cid - 1]
                    
                    if not (sb <= p < eb):
                        local = total_gauss[sb:eb]
                        if len(local) == 0:
                            continue
                        p = sb + int(np.argmax(local))
                    
                    ws = max(sb, p - W_LEFT)
                    we = min(eb - 1, p + W_RIGHT)
                    
                    keep_mask[ws:we + 1] = True
                
                df_peak_window = df_label[keep_mask].copy()
                st.success(f"Peak-window rows: {len(df_peak_window)} | Cycles: {df_peak_window['cycle_id'].nunique()}")
                
                # Plot
                fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
                
                axes[0].plot(x, total_raw, alpha=0.3, label="Total raw")
                axes[0].plot(x, total_gauss, label=f"Gaussian Total σ={GAUSS_SIGMA:g}")
                
                px = x.iloc[peaks] if use_time else x.to_numpy()[peaks]
                axes[0].scatter(px, total_gauss[peaks], s=50, label="Peaks")
                axes[0].legend()
                axes[0].set_title("Total Raw + Gaussian + Peaks")
                
                for cid, g in df_peak_window.groupby("cycle_id", sort=True):
                    axes[1].plot(x.loc[g.index], g[TOTAL_COL], marker="o", markersize=3, label=f"cycle {cid}")
                axes[1].set_title("Peak-Window Raw per Cycle")
                axes[1].legend(ncol=4, fontsize=8)
                
                plt.tight_layout()
                
                # Display plot directly
                st.pyplot(fig)
                plt.close(fig)
                
                # Action buttons
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 3])
                
                with col1:
                    confirm_final_btn = st.button("✅ Confirm", type="primary")
                with col2:
                    tune_btn = st.button("🎛️ Tune", type="secondary")
                
                if confirm_final_btn:
                    # Save peak window data (in memory only)
                    st.session_state['final_confirmed'] = True
                    st.success("✅ บันทึกไฟล์")

                    # --- RADAR Visualization ---
                    radar_fig = None
                    try:
                        num_cycles = len(peaks)
                        ncols = 5
                        nrows = math.ceil(num_cycles / ncols)
                        radar_fig, _ = radar_peak_window(
                            df_peak_window,
                            SENSOR_COLS,
                            max_radar=num_cycles,
                            nrows=nrows,
                            ncols=ncols,
                            normalize=True
                        )
                        if radar_fig is not None:
                            st.markdown('---')
                            st.subheader('Radar Plot: Raw Peak-Window (mean per cycle)')
                            st.pyplot(radar_fig)
                            plt.close(radar_fig)
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการสร้าง Radar Plot: {e}")

                    # --- PCA Consistency Visualization ---
                    pca_fig = None
                    pca_loading = None
                    pca_labels = None
                    try:
                        st.markdown('---')
                        st.subheader('PCA Consistency Plot (Peak-Window)')
                        pca_fig, pca_loading, pca_labels = pca_consistency_plot(
                            df_peak_window,
                            SENSOR_COLS,
                            fingerprint="mean",
                            outlier_std=0.5,
                            figsize=(9, 7)
                        )
                        st.pyplot(pca_fig)
                        plt.close(pca_fig)
                        # Optionally show loading matrix and outlier labels
                        with st.expander("PCA Loadings & Outlier Labels"):
                            st.write("**PCA Loadings:**")
                            st.dataframe(pca_loading)
                            st.write("**Cycle Outlier Labels:**")
                            outlier_df = df_peak_window.groupby("cycle_id").size().reset_index(name="count")
                            outlier_df["label"] = pca_labels
                            st.dataframe(outlier_df[["cycle_id", "label"]])
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการสร้าง PCA Consistency Plot: {e}")

                    # --- Box Plot Visualization ---
                    box_fig = None
                    try:
                        st.markdown('---')
                        st.subheader('Box Plot: Sensor Value Distribution (Peak-Window)')
                        box_fig = box_plot(
                            df_peak_window,
                            SENSOR_COLS,
                            figsize=(10, 6)
                        )
                        st.pyplot(box_fig)
                        plt.close(box_fig)
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการสร้าง Box Plot: {e}")

                    # --- Download ZIP Button ---
                    import zipfile
                    import base64

                    def fig_to_png_bytes(fig):
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        buf.seek(0)
                        return buf.read()

                    # Prepare files in memory
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        # PDF Report
                        chemical_name = st.session_state.get('chemical_name', '')
                        figs_for_report = [fig, radar_fig, pca_fig, box_fig]
                        pdf_buf = create_report_package(chemical_name, figs_for_report)
                        pdf_buf.seek(0)
                        zf.writestr("report.pdf", pdf_buf.read())

                        # 1. all_data.csv
                        zf.writestr("all_data.csv", st.session_state['all_data.csv'])
                        # 2. peak_window_raw_all_data.csv
                        zf.writestr("peak_window_raw_all_data.csv", df_peak_window.to_csv(index=False))
                        # 3. total_peaks.png (2-plot figure)
                        if fig is not None:
                            zf.writestr("total_peaks.png", fig_to_png_bytes(fig))
                        # 4. radar_plot.png
                        if radar_fig is not None:
                            zf.writestr("radar_plot.png", fig_to_png_bytes(radar_fig))
                        # 5. pca_plot.png
                        if pca_fig is not None:
                            zf.writestr("pca_plot.png", fig_to_png_bytes(pca_fig))
                        # 6. box_plot.png
                        if box_fig is not None:
                            zf.writestr("box_plot.png", fig_to_png_bytes(box_fig))
                        # 7. pca_loadings.csv
                        if pca_loading is not None:
                            zf.writestr("pca_loadings.csv", pca_loading.to_csv())
                        # 8. outlier_labels.csv
                        if pca_labels is not None:
                            outlier_df = df_peak_window.groupby("cycle_id").size().reset_index(name="count")
                            outlier_df["label"] = pca_labels
                            zf.writestr("outlier_labels.csv", outlier_df[["cycle_id", "label"]].to_csv(index=False))
                    zip_buffer.seek(0)

                    st.markdown('---')
                    st.subheader('📦 Download All Output as ZIP')
                    # Orange button style for download
                    st.markdown("""
                        <style>
                        .stDownloadButton>button {
                            background-color: #ff9800;
                            color: white;
                        }
                        .stDownloadButton>button:hover {
                            background-color: #fb8c00;
                            color: white;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    st.download_button(
                        label="Download All Results (ZIP)",
                        data=zip_buffer,
                        file_name="smell_label_outputs.zip",
                        mime="application/zip"
                    )
                
                if tune_btn:
                    st.session_state['tuning_mode'] = True
                    st.rerun()
                
    except Exception as e:
        st.error(f"❌ Error in labeling process: {e}")
        import traceback
        st.code(traceback.format_exc())