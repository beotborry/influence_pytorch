# influence_pytorch

## Todo

1. s_test를 근사하는것이 아니라 hessian inverse 한번 구해놓고 계속 활용
hessian inverse => Fisher 이용??

=> hessian inverse를 구하는 시간이 더 소모

1. reweighting 보다 좋은 점을 보이기 위한 실험 
    => 차별을 할 수 있는 task를 가지고 Naive/reweighting 비교
    => 실험 setting 생각...
example 별로 weighting 줌, weight에 의미가 있음

1. influence를 이용 학습에 사용한 paper 찾기 => 없는듯

1. UTKFace 이용해서 유사성 찾기?

1. DP의 경우 특수한 상황에서만 loss_diff이 적용가능 이를 해결하는 방안?

1. train violation과 test violation 관계 보이기
    이건 그냥 train시 찍어보면 됨