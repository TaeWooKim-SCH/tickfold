import pytest

# 모델용 의존성 그룹(model)을 안 깐 환경에서는 이 파일을 건너뛴다
pytest.importorskip("torch")

import torch

from tickfold.model.deeplob import DeepLOB, parameter_count
from tickfold.model.settings import Settings


def test_output_is_three_logits_and_only_the_level_folding_kernel_grows_with_the_level_count():
    settings, counts = Settings(), {}
    for levels in (10, 20):
        model = DeepLOB(levels, settings.conv_filters, settings.inception_filters, settings.lstm_units)
        windows = torch.zeros(2, 1, 100, 4 * levels)
        # 시간 축을 채워 창 100행이 그대로 LSTM 으로 간다
        assert model.convolutions(windows).shape == (2, 16, 100, 1)
        assert model(windows).shape == (2, 3)
        counts[levels] = parameter_count(model)
    # 원 논문의 10레벨이 "약 60,000" 이다. 20레벨은 (1×N) 커널 16×16 이 10칸 늘어난 만큼만 많다
    assert counts == {10: 60_947, 20: 60_947 + 16 * 16 * 10}