import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
import json
import google.generativeai as genai

# 1. 页面基本配置
st.set_page_config(page_title="Etsy 订单分类解析工具", layout="wide")
st.title("📄 Etsy 订单 PDF 自动分类转 CSV (终极完整版)")

# 2. 获取 API Key
api_key = st.text_input("请输入 Google Gemini API Key:", type="password")

# 3. 文件上传组件
uploaded_file = st.file_uploader("请上传 Etsy 订单 PDF 文件", type="pdf")

if uploaded_file is not None and api_key:
    if st.button("开始解析订单"):
        # 将 try 放在最外层，完美捕捉所有可能的错误
        try:
            # ================= 阶段 1：AI 结构化提取 =================
            with st.spinner('⏳ 阶段 1/2：正在读取 PDF 并调用 Gemini 分析数据...'):
                # 步骤 A: 提取 PDF 文本
                full_text = ""
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages:
                        full_text += page.extract_text() + "\n"
                
                # 步骤 B: 配置并调用 Gemini API
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
                
                # 带有强力负面指令的 Prompt
                prompt = """
                你是一个高级订单数据提取专家。请仔细阅读以下 Etsy 订单发货单文本，将其解析并分类为 JSON 格式。
                输出必须是包含两个数组的 JSON 对象："decanters" 和 "bags"。
                
                1. "decanters" 数组包含 Decanter/Whiskey 相关商品，字段：OrderNumber, Design, Initial, Name, Title, Date。
                2. "bags" 数组包含 Bag 相关商品，字段：OrderNumber, Font, Name。
                
                【极其重要的提取规则 - 请严格遵守】：
                - Date (雕刻日期)：必须且只能从客户的 "Personalization:" 备注文本中提取客户明确要求刻在产品上的纪念日/日期（例如 "10.20.2026", "5-9-2026"）。
                  🚫 **绝对禁止**提取页面上的 "Order date" (下单日期) 或 "Scheduled to ship by" (发货日期)。如果客户的定制备注中没有提到任何日期，此字段**必须严格留空 ""**。
                - Initial (首字母)：如果定制备注中没有明确写明需要哪个首字母，强制留空 ""。
                - 遇到合并数量的定制信息，请正确拆分为多行，并将订单号、公共属性复制给每一行。
                
                订单文本如下：
                """
                response = model.generate_content(prompt + full_text)
                
                # 步骤 C: 解析 JSON 结果
                result_data = json.loads(response.text)
                decanters_data = result_data.get("decanters", [])
                bags_data = result_data.get("bags", [])
                    
            # ================= 阶段 2：Pandas 逻辑清洗 =================
            with st.spinner('⚙️ 阶段 2/2：正在执行严格的格式清洗与序号重置...'):
                
                # --- 处理 Decanters 数据 ---
                df_decanters = pd.DataFrame()
                if decanters_data:
                    df_decanters = pd.DataFrame(decanters_data)
                    
                    # 清洗 1：提取 Design 纯数字 (例如 "Design #3" -> "3")
                    if 'Design' in df_decanters.columns:
                        df_decanters['Design'] = df_decanters['Design'].astype(str).str.extract(r'(\d+)')[0]
                    
                    # 清洗 2：处理 Name 和 Initial 的前后空格
                    if 'Name' in df_decanters.columns:
                        df_decanters['Name'] = df_decanters['Name'].astype(str).str.strip()
                    if 'Initial' in df_decanters.columns:
                        df_decanters['Initial'] = df_decanters['Initial'].astype(str).str.strip()
                    else:
                        df_decanters['Initial'] = pd.NA
                    
                    # 清洗 3：获取 Name 的大写首字母作为备用 Initial
                    derived_initial = df_decanters['Name'].str[0].str.upper()
                    
                    # 如果 Initial 为空、"None" 或 NaN，则填入备用 Initial
                    df_decanters['Initial'] = df_decanters['Initial'].replace(['', 'nan', 'None', 'NaN'], pd.NA).fillna(derived_initial)
                    
                    # 清洗 4：专门处理 Design 为 1 的情况，将 Design 改为 1 + Initial
                    mask_design_1 = df_decanters['Design'] == '1'
                    df_decanters.loc[mask_design_1, 'Design'] = '1' + df_decanters.loc[mask_design_1, 'Initial'].astype(str)
                    
                    # 清洗 5：按订单号重新生成 1 起始的 Row 序号
                    df_decanters['Row'] = df_decanters.groupby('OrderNumber').cumcount() + 1
                    
                    # 重新排列列的顺序，确保存在
                    cols = ['Row', 'OrderNumber', 'Design', 'Initial', 'Name', 'Title', 'Date']
                    df_decanters = df_decanters[[c for c in cols if c in df_decanters.columns]]


                # --- 处理 Bags 数据 ---
                df_bags = pd.DataFrame()
                if bags_data:
                    df_bags = pd.DataFrame(bags_data)
                    
                    # 清洗 1：提取 Font 纯数字 (例如 "Font #1" -> "1")
                    if 'Font' in df_bags.columns:
                        df_bags['Font'] = df_bags['Font'].astype(str).str.extract(r'(\d+)')[0]
                        
                    # 清洗 2：为 Bag 重置 Row 序号
                    df_bags['Row'] = df_bags.groupby('OrderNumber').cumcount() + 1
                    
                    # 重排列顺序
                    cols_bags = ['Row', 'OrderNumber', 'Font', 'Name']
                    df_bags = df_bags[[c for c in cols_bags if c in df_bags.columns]]


            # ================= 渲染结果 =================
            st.success("✅ 解析与清洗成功！")
            
            # 步骤 D: 在网页上分块展示并提供下载
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🥃 醒酒器 (Decanters) 数据")
                if not df_decanters.empty:
                    st.dataframe(df_decanters, use_container_width=True)
                    csv_decanters = df_decanters.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(label="📥 下载 Decanters CSV", data=csv_decanters, file_name="decanters_orders.csv", mime="text/csv", key="download_decanters")
                else:
                    st.info("该 PDF 中未检测到 Decanter 订单。")
                    
            with col2:
                st.subheader("👜 伴娘包 (Bags) 数据")
                if not df_bags.empty:
                    st.dataframe(df_bags, use_container_width=True)
                    csv_bags = df_bags.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(label="📥 下载 Bags CSV", data=csv_bags, file_name="bags_orders.csv", mime="text/csv", key="download_bags")
                else:
                    st.info("该 PDF 中未检测到 Bag 订单。")
                    
        except Exception as e:
            st.error(f"处理过程中出现错误: {e}")