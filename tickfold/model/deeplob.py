"""DeepLOB(Zhang, Zohren, Roberts 2019)을 레벨 수 N 을 인자로 받게 쓴 것. 공개 구현은 라이선스 표기가 없어 논문 그림 3·4 와 4절을 보고 썼다."""

import torch
from torch import nn


class DeepLOB(nn.Module):
    """입력 (배치, 1, 창, 4N) 을 내림·보합·오름 세 로짓으로. 열은 레벨마다 매도가, 매도량, 매수가, 매수량이다(논문 식 6).

    (1×2, 보폭 1×2) 둘이 (가격, 수량) 과 (매도, 매수) 를 묶고 나면 레벨 N 칸이 남아, 세 번째 블록 첫 커널을 (1×N) 으로 둔다.
    논문은 N=10 이다. 이 커널만 N 따라 커져서 파라미터 수가 달라지므로 결과에 같이 싣는다. 논문대로 모든 합성곱은 시간 축을
    0 으로 채워 창 길이를 지키고, 활성 함수는 Leaky-ReLU(0.01) 하나다. 논문에 없는 배치 정규화는 넣지 않는다.
    """

    def __init__(self, levels: int, conv_filters: int, inception_filters: int, lstm_units: int):
        super().__init__()
        self.convolutions = nn.Sequential(
            _block(1, conv_filters, (1, 2), (1, 2)),
            _block(conv_filters, conv_filters, (1, 2), (1, 2)),
            _block(conv_filters, conv_filters, (1, levels), (1, 1)),
        )
        # 인셉션 모듈은 시간 폭이 다른 세 갈래다. 이동평균을 여러 길이로 겹쳐 보는 것과 같다고 논문이 설명한다
        self.inception_short = _inception_branch(conv_filters, inception_filters, 3)
        self.inception_long = _inception_branch(conv_filters, inception_filters, 5)
        pooling = nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0))
        self.inception_pool = nn.Sequential(pooling, _convolution(conv_filters, inception_filters, (1, 1)))
        self.lstm = nn.LSTM(3 * inception_filters, lstm_units, batch_first=True)
        self.output = nn.Linear(lstm_units, 3)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """로짓을 돌려준다. 확률은 부르는 쪽이 softmax 로 낸다."""
        features = self.convolutions(windows)  # (배치, 필터, 창, 1)
        branches = (self.inception_short(features), self.inception_long(features), self.inception_pool(features))
        sequence = torch.cat(branches, dim=1).squeeze(3).transpose(1, 2)  # (배치, 창, 3 × 인셉션 필터)
        outputs, _ = self.lstm(sequence)
        return self.output(outputs[:, -1])


def parameter_count(model: nn.Module) -> int:
    """학습하는 파라미터 수. 깊이 20 과 10 의 용량 차이를 결과 표에 같이 적는다."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _block(in_channels: int, out_channels: int, kernel: tuple[int, int], stride: tuple[int, int]) -> nn.Sequential:
    # 레벨 축을 접는 합성곱 하나 뒤에 시간 축 (4×1) 둘. 논문 그림 3 의 한 줄이다
    return nn.Sequential(
        _convolution(in_channels, out_channels, kernel, stride),
        _convolution(out_channels, out_channels, (4, 1)),
        _convolution(out_channels, out_channels, (4, 1)),
    )


def _inception_branch(in_channels: int, out_channels: int, height: int) -> nn.Sequential:
    # (1×1) 로 채널을 바꾼 뒤 시간 폭 height 로 본다. 논문 그림 4 의 한 갈래다
    return nn.Sequential(
        _convolution(in_channels, out_channels, (1, 1)), _convolution(out_channels, out_channels, (height, 1))
    )


def _convolution(in_channels, out_channels, kernel, stride=(1, 1)) -> nn.Sequential:
    # 시간 축만 채운다. 레벨 축을 채우면 접은 폭이 다시 늘어난다. 짝수 커널은 Keras 의 same 처럼 뒤를 하나 더 채운다
    before = (kernel[0] - 1) // 2
    padding = [nn.ZeroPad2d((0, 0, before, kernel[0] - 1 - before))] if kernel[0] > 1 else []
    return nn.Sequential(*padding, nn.Conv2d(in_channels, out_channels, kernel, stride), nn.LeakyReLU(0.01))