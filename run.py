import os
import shutil
import asyncio
from sentence_transformers import SentenceTransformer, util
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QTextEdit, QProgressBar, QLabel, QLineEdit
)
from PyQt5.QtCore import Qt
from qasync import QEventLoop


class IntelligentFileOrganizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("유사 파일 및 폴더 정리 도우미")
        self.setGeometry(100, 100, 900, 700)

        # UI 설정
        self.setup_ui()

        # NLP 및 Multi-Model 초기화
        self.text_model = self.load_text_model()
        #self.summary_pipeline = self.load_summarization_model()

        # 선택된 폴더 경로 및 상태 데이터
        self.selected_folder = None
        self.folder_similarities = {}  # 유사 폴더 분석 결과
        self.file_groups = {}  # 파일 유사도 그룹화 결과
        self.simulated_organization = []  # 시뮬레이션된 정리 결과

    def setup_ui(self):
        self.title_label = QLabel("유사 파일 및 폴더 정리 도우미", self)
        self.title_label.setGeometry(200, 10, 500, 40)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        self.select_folder_button = QPushButton("폴더 선택", self)
        self.select_folder_button.setGeometry(50, 60, 200, 40)
        self.select_folder_button.clicked.connect(self.select_folder)

        self.analyze_folders_button = QPushButton("폴더 분석", self)
        self.analyze_folders_button.setGeometry(50, 160, 200, 40)
        self.analyze_folders_button.clicked.connect(lambda: asyncio.ensure_future(self.analyze_similar_folders()))

        self.organize_folders_button = QPushButton("폴더 정리", self)
        self.organize_folders_button.setGeometry(270, 160, 200, 40)
        self.organize_folders_button.clicked.connect(lambda: asyncio.ensure_future(self.group_similar_folders()))

        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("파일 또는 폴더 검색...")
        self.search_input.setGeometry(50, 220, 300, 40)

        self.search_button = QPushButton("검색", self)
        self.search_button.setGeometry(370, 220, 100, 40)
        self.search_button.clicked.connect(lambda: asyncio.ensure_future(self.search_files_and_folders()))

        self.result_area = QTextEdit(self)
        self.result_area.setGeometry(50, 280, 800, 300)
        self.result_area.setReadOnly(True)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(50, 630, 800, 30)
        self.progress_bar.setValue(0)

    def load_text_model(self):
        """
        SentenceTransformer 모델을 로컬에서 로드하거나, 없으면 다운로드 후 로드
        """
        model_cache_dir = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(model_cache_dir, exist_ok=True)
        model_name = "all-MiniLM-L6-v2"
        model_path = os.path.join(model_cache_dir, model_name)

        if not os.path.exists(model_path):
            self.result_area.append("SentenceTransformer 모델 다운로드 중...")
            model = SentenceTransformer(model_name)
            model.save(model_path)
            self.result_area.append("모델 다운로드 완료. 로컬에 저장되었습니다.")
        else:
            self.result_area.append("SentenceTransformer 모델을 로컬에서 로드합니다.")
            model = SentenceTransformer(model_path)

        return model

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if folder:
            self.selected_folder = folder
            self.result_area.append(f"선택된 폴더: {folder}")

    async def analyze_similar_folders(self):
        """
        폴더와 파일의 유사도를 분석하고, 예상 정리 결과를 표시
        """
        if not self.selected_folder:
            self.result_area.append("먼저 폴더를 선택하세요!")
            return

        # 초기화
        self.simulated_organization = []
        self.progress_bar.setValue(0)

        # 폴더 이름 유사도 분석
        subfolders = [f.path for f in os.scandir(self.selected_folder) if f.is_dir()]
        if subfolders:
            self.result_area.append(f"총 {len(subfolders)}개의 폴더를 분석 중...")
            folder_names = [os.path.basename(folder) for folder in subfolders]

            # 임베딩 계산 (비동기로 처리)
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(None, lambda: self.text_model.encode(folder_names, convert_to_tensor=True))

            self.folder_groups = {}
            grouped_folders = set()

            for i in range(len(folder_names)):
                if subfolders[i] in grouped_folders:
                    continue

                max_similarity = 0
                best_group = None

                for group_key, group_folders in self.folder_groups.items():
                    group_embeddings = [embeddings[folder_names.index(os.path.basename(f))] for f in group_folders]
                    avg_similarity = sum(util.pytorch_cos_sim(embeddings[i], emb).item() for emb in group_embeddings) / len(group_embeddings)

                    if avg_similarity > max_similarity:
                        max_similarity = avg_similarity
                        best_group = group_key

                if max_similarity > 0.3 and best_group:
                    self.folder_groups[best_group].append(subfolders[i])
                    grouped_folders.add(subfolders[i])
                else:
                    self.folder_groups[subfolders[i]] = [subfolders[i]]
                    grouped_folders.add(subfolders[i])

            # 분석 결과 표시
            self.result_area.append("\n폴더 분석 결과:")
            for folder, similar_folders in self.folder_groups.items():
                self.result_area.append(f" - {folder} -> 유사 폴더: {', '.join(similar_folders)}")

        else:
            self.result_area.append("유사 폴더가 발견되지 않았습니다.")

        # 파일 이름 및 내용 유사도 분석
        files = [f for f in os.listdir(self.selected_folder) if os.path.isfile(os.path.join(self.selected_folder, f))]
        if files:
            self.result_area.append(f"총 {len(files)}개의 파일을 분석 중...")
            file_names = [os.path.splitext(f)[0] for f in files]

            # 임베딩 계산 (비동기로 처리)
            embeddings = await loop.run_in_executor(None, lambda: self.text_model.encode(file_names, convert_to_tensor=True))

            self.file_groups = {}
            grouped_files = set()

            for i in range(len(file_names)):
                if files[i] in grouped_files:
                    continue

                max_similarity = 0
                best_group = None

                for group_key, group_files in self.file_groups.items():
                    group_embeddings = [embeddings[file_names.index(os.path.splitext(f)[0])] for f in group_files]
                    avg_similarity = sum(util.pytorch_cos_sim(embeddings[i], emb).item() for emb in group_embeddings) / len(group_embeddings)

                    if avg_similarity > max_similarity:
                        max_similarity = avg_similarity
                        best_group = group_key

                if max_similarity > 0.5 and best_group:
                    self.file_groups[best_group].append(files[i])
                    grouped_files.add(files[i])
                else:
                    self.file_groups[files[i]] = [files[i]]
                    grouped_files.add(files[i])

            # 파일 그룹화 결과 표시
            self.result_area.append("\n파일 유사도 그룹화 결과:")
            for group_name, group_files in self.file_groups.items():
                self.result_area.append(f" - {group_name}와 유사한 파일: {', '.join(group_files)}")

        else:
            self.result_area.append("유사 파일이 발견되지 않았습니다.")

        # UI 업데이트
        self.progress_bar.setValue(100)
        self.result_area.append("분석이 완료되었습니다.")




    def get_representative_name(self, names):
        """
        유사 그룹에서 대표 이름을 선정
        """
        return min(names, key=lambda name: (len(name), -sum(1 for other in names if name in other)))

    async def group_similar_folders(self):
        """
        폴더와 파일을 실제로 정리
        """
        self.progress_bar.setValue(0)

        if not self.selected_folder:
            self.result_area.append("먼저 폴더를 선택하세요!")
            return

        self.result_area.append("유사 폴더 및 파일 그룹화를 시작합니다...")

        # 파일 유사도 기반 그룹화 정리
        if self.file_groups:
            for group_name, group_files in self.file_groups.items():
                group_folder = self._get_unique_folder_path(os.path.join(self.selected_folder, group_name))
                os.makedirs(group_folder, exist_ok=True)

                for file in group_files:
                    src_path = os.path.join(self.selected_folder, file)
                    dest_path = os.path.join(group_folder, file)
                    dest_path = self._get_unique_file_path(dest_path)
                    await self.async_move(src_path, dest_path)

        # 폴더 유사도 기반 그룹화 정리
        if self.folder_groups:
            total_groups = len(self.folder_groups)
            group_idx = 0

            for group_key, group_folders in self.folder_groups.items():
                all_folders = [group_key] + group_folders
                representative_name = self.get_representative_name([os.path.basename(f) for f in all_folders])
                group_folder = self._get_unique_folder_path(os.path.join(self.selected_folder, representative_name))
                os.makedirs(group_folder, exist_ok=True)

                for folder in all_folders:
                    dest_folder = os.path.join(group_folder, os.path.basename(folder))
                    await self._move_folder_contents(folder, dest_folder)

                group_idx += 1
                progress = int((group_idx / total_groups) * 100)
                self.progress_bar.setValue(progress)
                self.result_area.append(f"'{representative_name}' 폴더로 그룹 정리 완료.")

            self.result_area.append("유사 폴더 및 파일 그룹화 정리가 완료되었습니다.")
        else:
            self.result_area.append("유사 폴더 분석 결과가 없습니다. 먼저 분석을 실행하세요.")

        self.progress_bar.setValue(100)
        
    async def search_files_and_folders(self):
        """
        입력된 키워드로 파일 및 폴더를 검색
        """
        if not self.selected_folder:
            self.result_area.append("먼저 폴더를 선택하세요!")
            return

        search_query = self.search_input.text().strip()
        if not search_query:
            self.result_area.append("검색어를 입력하세요!")
            return

        self.result_area.append(f"'{search_query}'로 검색 중...")
        all_items = [f.path for f in os.scandir(self.selected_folder)]
        item_names = [os.path.basename(item) for item in all_items]
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, lambda:self.text_model.encode(item_names, convert_to_tensor=True))
        query_embedding = await loop.run_in_executor(None, lambda:self.text_model.encode(search_query, convert_to_tensor=True))

        self.search_results = []
        for idx, item in enumerate(all_items):
            similarity = util.pytorch_cos_sim(query_embedding, embeddings[idx]).item()
            if similarity > 0.5:  # 검색 유사도 임계값
                self.search_results.append((item, similarity))

        self.search_results.sort(key=lambda x: x[1], reverse=True)

        self.result_area.append("\n검색 결과:")
        if self.search_results:
            for item, sim in self.search_results:
                self.result_area.append(f" - {item} (유사도: {sim:.2f})")
        else:
            self.result_area.append("검색 결과가 없습니다.")

    async def _move_folder_contents(self, src_folder, dest_folder):
        """
        폴더의 모든 내용물을 재귀적으로 이동
        """
        if not os.path.exists(src_folder):
            self.result_area.append(f"'{src_folder}' 폴더를 찾을 수 없습니다. 스킵합니다.")
            return

        for item in os.listdir(src_folder):
            src_path = os.path.join(src_folder, item)
            dest_path = os.path.join(dest_folder, item)

            if os.path.isdir(src_path):
                os.makedirs(dest_path, exist_ok=True)
                await self._move_folder_contents(src_path, dest_path)
            else:
                dest_path = self._get_unique_file_path(dest_path)
                if os.path.exists(src_path):
                    try:
                        await self.async_move(src_path, dest_path)
                    except FileNotFoundError:
                        self.result_area.append(f"에러: 파일을 찾을 수 없음 -> {src_path}")
                else:
                    self.result_area.append(f"파일 없음, 스킵됨: {src_path}")

        if os.path.exists(src_folder) and not os.listdir(src_folder):
            os.rmdir(src_folder)

    def _get_unique_file_path(self, file_path):
        base, ext = os.path.splitext(file_path)
        counter = 1
        while os.path.exists(file_path):
            file_path = f"{base}_{counter}{ext}"
            counter += 1
        return file_path

    def _get_unique_folder_path(self, folder_path):
        base_path = folder_path
        counter = 1
        while os.path.exists(folder_path):
            folder_path = f"{base_path}_{counter}"
            counter += 1
        return folder_path

    async def async_move(self, src, dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        try:
            self.result_area.append(f"파일 이동: {src} -> {dest}")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda:shutil.move(src, dest))
        except FileNotFoundError:
            self.result_area.append(f"에러: '{src}' 파일이 존재하지 않습니다. 이동 스킵됨.")
        except Exception as e:
            self.result_area.append(f"에러: {src} -> {dest}, {e}")
            

if __name__ == "__main__":
    app = QApplication([])
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = IntelligentFileOrganizer()
    window.show()

    with loop:
        loop.run_forever()