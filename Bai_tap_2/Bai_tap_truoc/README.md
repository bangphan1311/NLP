# Vietnamese word embedding (CBOW tự cài đặt)

Bài tập số 2 môn Xử lý ngôn ngữ tự nhiên: tự huấn luyện word embedding cho tiếng Việt
bằng CBOW (PyTorch, không dùng gensim hay mô hình có sẵn), so sánh ba hàm mất mát
(naive softmax, hierarchical softmax, negative sampling) và so sánh corpus tách từ
với không tách từ.

## Cấu trúc
- `src/prepare_data.py`, `clean_corpus.py`, `create_two_corpora.py`, `build_vocab.py`, `prepare_cbow.py`: tiền xử lý
- `src/cbow_model.py`, `cbow_hierarchical_softmax_model.py`, `cbow_negative_sampling_model.py`: ba mô hình
- `src/train_compare.py`: huấn luyện và so sánh các hàm mất mát, tách từ / không tách từ
- `src/train_cbow_segmented_full.py`: lượt tiếp tục epoch 11-30 của mô hình chính
- `src/evaluate_compare.py`, `evaluate_tokenization.py`, `eval_words.py`, `visualize.py`: đánh giá và hình
- `results/`: loss, bảng so sánh, hình PCA/t-SNE
- `BAO_CAO.txt`: mô tả cách làm và kết quả

## Chạy lại
    uv sync
    uv run python src/prepare_data.py
    uv run python src/clean_corpus.py
    uv run python src/create_two_corpora.py
    uv run python src/build_vocab.py
    uv run python src/prepare_cbow.py
    uv run python src/train_compare.py --loss ns --corpus segmented --num-docs 10000 --epochs 5
    uv run python src/train_compare.py --loss hs --corpus segmented --num-docs 10000 --epochs 5
    uv run python src/train_compare.py --loss softmax --corpus segmented --num-docs 10000 --epochs 5
    uv run python src/train_compare.py --loss ns --corpus unsegmented --num-docs 10000 --epochs 5
    uv run python src/evaluate_compare.py
    uv run python src/evaluate_tokenization.py
    uv run python src/visualize.py

## Không có trong repo
Thư mục `data/` và `models/` (dung lượng lớn) được tạo lại bằng các bước trên.
Checkpoint chính 30 epoch không tái lập hoàn toàn từ repo (xem `BAO_CAO.txt`, mục 8).