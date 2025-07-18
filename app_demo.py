import streamlit as st
import numpy as np
import cv2
import sqlite3
import os
import sys
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pickle
import base64
from io import BytesIO
import tempfile

# Cấu hình trang
st.set_page_config(
    page_title="AI Signature Verification - Demo",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .demo-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SimpleImageProcessor:
    """Processor đơn giản cho demo cloud"""
    
    @staticmethod
    def process_signature(image):
        """Xử lý ảnh chữ ký cơ bản"""
        try:
            # Chuyển về grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Threshold để tách nền
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Tìm contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Tìm bounding box lớn nhất
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Crop và padding
                margin = 20
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(gray.shape[1] - x, w + 2*margin)
                h = min(gray.shape[0] - y, h + 2*margin)
                
                cropped = gray[y:y+h, x:x+w]
            else:
                cropped = gray
            
            # Resize về kích thước chuẩn
            resized = cv2.resize(cropped, (128, 128))
            
            # Normalize
            normalized = resized.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            st.error(f"Lỗi xử lý ảnh: {str(e)}")
            return None

class SimpleDatabaseDemo:
    """Database đơn giản trong memory cho demo"""
    
    def __init__(self):
        self.users = {"Demo User": {"id": 1, "email": "demo@example.com"}}
        self.signatures = []
        self.verifications = []
    
    def add_signature(self, user_name, image_data, is_template=True):
        """Thêm chữ ký"""
        sig_id = len(self.signatures) + 1
        self.signatures.append({
            "id": sig_id,
            "user": user_name,
            "data": image_data,
            "is_template": is_template
        })
        return sig_id
    
    def get_user_templates(self, user_name):
        """Lấy chữ ký mẫu của user"""
        return [sig for sig in self.signatures if sig["user"] == user_name and sig["is_template"]]
    
    def add_verification(self, user_name, similarity, is_genuine):
        """Thêm kết quả xác minh"""
        self.verifications.append({
            "user": user_name,
            "similarity": similarity,
            "is_genuine": is_genuine
        })

class SignatureAppDemo:
    def __init__(self):
        self.processor = SimpleImageProcessor()
        self.db = SimpleDatabaseDemo()
        
        # Khởi tạo session state
        if 'demo_user' not in st.session_state:
            st.session_state.demo_user = "Demo User"
        if 'templates' not in st.session_state:
            st.session_state.templates = []
    
    def main(self):
        # Header
        st.markdown('<h1 class="main-header">🖋️ AI Signature Verification - Demo</h1>', unsafe_allow_html=True)
        
        # Demo warning
        st.markdown("""
        <div class="demo-warning">
            <h4>🌟 Đây Là Phiên Bản Demo</h4>
            <p>• Dữ liệu chỉ lưu tạm thời trong session</p>
            <p>• Tính năng đơn giản hóa cho cloud deployment</p>
            <p>• Để trải nghiệm đầy đủ, hãy tải app về máy</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar menu
        st.sidebar.title("📋 Menu")
        menu = st.sidebar.selectbox(
            "Chọn chức năng:",
            ["🏠 Giới thiệu", "📝 Upload Chữ Ký Mẫu", "🔍 Xác Minh Chữ Ký", "🎨 Vẽ Chữ Ký"]
        )
        
        if menu == "🏠 Giới thiệu":
            self.intro_page()
        elif menu == "📝 Upload Chữ Ký Mẫu":
            self.upload_signature()
        elif menu == "🔍 Xác Minh Chữ Ký":
            self.verify_demo()
        elif menu == "🎨 Vẽ Chữ Ký":
            self.draw_signature()
    
    def intro_page(self):
        st.header("🎯 Giới Thiệu Ứng Dụng")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✨ Tính Năng Chính")
            st.write("""
            - 🔍 **Xác minh chữ ký**: Phân biệt chữ ký thật/giả
            - 📝 **Đăng ký mẫu**: Upload chữ ký để so sánh
            - 🎨 **Vẽ trực tiếp**: Vẽ chữ ký trên canvas
            - 📊 **Phân tích**: Hiển thị độ tương đồng
            """)
            
            st.subheader("🔬 Công Nghệ")
            st.write("""
            - **AI**: Siamese Neural Network
            - **Computer Vision**: OpenCV
            - **Framework**: Streamlit + TensorFlow
            - **Database**: SQLite
            """)
        
        with col2:
            st.subheader("🚀 Hướng Dẫn Nhanh")
            st.write("""
            **Bước 1**: Upload chữ ký mẫu (tab "Upload Chữ Ký Mẫu")
            
            **Bước 2**: Upload ảnh cần xác minh (tab "Xác Minh Chữ Ký")
            
            **Bước 3**: Xem kết quả phân tích
            
            **Hoặc**: Vẽ chữ ký trực tiếp trên canvas
            """)
            
            st.info("💡 **Mẹo**: Ảnh chữ ký nên có nền trắng, chữ đen để kết quả tốt nhất!")
    
    def upload_signature(self):
        st.header("📝 Upload Chữ Ký Mẫu")
        
        st.write("Upload ảnh chữ ký để làm mẫu so sánh:")
        
        uploaded_file = st.file_uploader(
            "Chọn file ảnh chữ ký:",
            type=['png', 'jpg', 'jpeg'],
            help="Chọn ảnh có nền trắng, chữ ký rõ nét"
        )
        
        if uploaded_file is not None:
            # Hiển thị ảnh gốc
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Ảnh Gốc")
                st.image(image, caption="Ảnh chữ ký đã upload")
            
            # Xử lý ảnh
            image_array = np.array(image)
            processed = self.processor.process_signature(image_array)
            
            if processed is not None:
                with col2:
                    st.subheader("Ảnh Đã Xử Lý")
                    st.image(processed, caption="Ảnh sau khi xử lý", cmap='gray')
                
                # Lưu vào session
                if st.button("💾 Lưu Làm Mẫu"):
                    template_id = self.db.add_signature("Demo User", processed, True)
                    st.session_state.templates.append({
                        "id": template_id,
                        "data": processed,
                        "name": f"Mẫu {len(st.session_state.templates) + 1}"
                    })
                    
                    st.markdown("""
                    <div class="success-box">
                        ✅ Đã lưu chữ ký mẫu thành công!
                    </div>
                    """, unsafe_allow_html=True)
        
        # Hiển thị danh sách mẫu đã lưu
        if st.session_state.templates:
            st.subheader("📋 Danh Sách Mẫu Đã Lưu")
            
            cols = st.columns(min(3, len(st.session_state.templates)))
            for i, template in enumerate(st.session_state.templates):
                with cols[i % 3]:
                    st.image(template["data"], caption=template["name"], width=150)
    
    def verify_demo(self):
        st.header("🔍 Xác Minh Chữ Ký")
        
        if not st.session_state.templates:
            st.warning("⚠️ Chưa có chữ ký mẫu! Vui lòng upload mẫu trước.")
            return
        
        st.write("Upload ảnh chữ ký cần xác minh:")
        
        uploaded_file = st.file_uploader(
            "Chọn file ảnh cần kiểm tra:",
            type=['png', 'jpg', 'jpeg'],
            key="verify_upload"
        )
        
        if uploaded_file is not None:
            # Hiển thị ảnh
            image = Image.open(uploaded_file)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Ảnh Cần Kiểm Tra")
                st.image(image, caption="Ảnh upload")
            
            # Xử lý ảnh
            image_array = np.array(image)
            processed = self.processor.process_signature(image_array)
            
            if processed is not None:
                with col2:
                    st.subheader("Ảnh Đã Xử Lý")
                    st.image(processed, caption="Sau xử lý", cmap='gray')
                
                # So sánh với mẫu (demo đơn giản)
                similarities = []
                for template in st.session_state.templates:
                    # Tính toán similarity đơn giản bằng correlation
                    correlation = np.corrcoef(processed.flatten(), template["data"].flatten())[0, 1]
                    # Chuyển về phần trăm
                    similarity = max(0, correlation * 100)
                    similarities.append(similarity)
                
                max_similarity = max(similarities) if similarities else 0
                is_genuine = max_similarity > 70  # Ngưỡng demo
                
                with col3:
                    st.subheader("Kết Quả")
                    
                    # Hiển thị kết quả
                    if is_genuine:
                        st.markdown(f"""
                        <div class="success-box">
                            <h4>✅ CHỮ KÝ HỢP LỆ</h4>
                            <p><strong>Độ tương đồng:</strong> {max_similarity:.1f}%</p>
                            <p><strong>Đánh giá:</strong> Chữ ký khớp với mẫu</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="danger-box">
                            <h4>❌ CHỮ KÝ KHÔNG HỢP LỆ</h4>
                            <p><strong>Độ tương đồng:</strong> {max_similarity:.1f}%</p>
                            <p><strong>Đánh giá:</strong> Chữ ký không khớp với mẫu</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Lưu kết quả
                self.db.add_verification("Demo User", max_similarity, is_genuine)
                
                # Hiển thị chi tiết so sánh
                st.subheader("📊 Chi Tiết So Sánh")
                for i, (template, sim) in enumerate(zip(st.session_state.templates, similarities)):
                    col_a, col_b, col_c = st.columns([1, 1, 2])
                    with col_a:
                        st.image(template["data"], caption=f"Mẫu {i+1}", width=100)
                    with col_b:
                        st.image(processed, caption="Ảnh test", width=100)
                    with col_c:
                        st.metric(f"Độ tương đồng với Mẫu {i+1}", f"{sim:.1f}%")
    
    def draw_signature(self):
        st.header("🎨 Vẽ Chữ Ký Trực Tiếp")
        
        st.write("Vẽ chữ ký của bạn trên canvas bên dưới:")
        
        # Tạo canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",  # Nền trong suốt
            stroke_width=3,
            stroke_color="#000000",
            background_color="#ffffff",
            height=200,
            width=600,
            drawing_mode="freedraw",
            key="signature_canvas",
        )
        
        if canvas_result.image_data is not None:
            # Chuyển đổi canvas data
            canvas_image = canvas_result.image_data.astype(np.uint8)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Lưu Làm Mẫu"):
                    processed = self.processor.process_signature(canvas_image)
                    if processed is not None:
                        template_id = self.db.add_signature("Demo User", processed, True)
                        st.session_state.templates.append({
                            "id": template_id,
                            "data": processed,
                            "name": f"Mẫu vẽ {len(st.session_state.templates) + 1}"
                        })
                        st.success("✅ Đã lưu chữ ký vẽ làm mẫu!")
            
            with col2:
                if st.button("🔍 Xác Minh Ngay"):
                    if st.session_state.templates:
                        processed = self.processor.process_signature(canvas_image)
                        if processed is not None:
                            # So sánh với mẫu
                            similarities = []
                            for template in st.session_state.templates:
                                correlation = np.corrcoef(processed.flatten(), template["data"].flatten())[0, 1]
                                similarity = max(0, correlation * 100)
                                similarities.append(similarity)
                            
                            max_similarity = max(similarities) if similarities else 0
                            is_genuine = max_similarity > 70
                            
                            if is_genuine:
                                st.success(f"✅ Chữ ký hợp lệ! Độ tương đồng: {max_similarity:.1f}%")
                            else:
                                st.error(f"❌ Chữ ký không hợp lệ! Độ tương đồng: {max_similarity:.1f}%")
                    else:
                        st.warning("⚠️ Chưa có mẫu để so sánh!")
            
            # Hiển thị ảnh đã xử lý
            if canvas_result.image_data is not None:
                processed = self.processor.process_signature(canvas_image)
                if processed is not None:
                    st.subheader("Ảnh Sau Xử Lý")
                    st.image(processed, caption="Chữ ký đã vẽ (sau xử lý)", width=300)

def main():
    try:
        app = SignatureAppDemo()
        app.main()
    except Exception as e:
        st.error(f"Lỗi ứng dụng: {str(e)}")
        st.write("Vui lòng tải lại trang hoặc liên hệ admin.")

if __name__ == "__main__":
    main()
