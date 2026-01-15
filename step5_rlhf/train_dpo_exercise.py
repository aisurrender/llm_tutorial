"""
Step 5: DPO 训练

练习文件：请完成标记为 TODO 的部分

DPO 核心公式：
L_DPO = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
"""

import torch
import torch.nn.functional as F


# =============================================================================
# TODO 1: 理解并实现 DPO Loss
# =============================================================================

def dpo_loss(
    policy_chosen_log_probs: torch.Tensor,
    policy_rejected_log_probs: torch.Tensor,
    ref_chosen_log_probs: torch.Tensor,
    ref_rejected_log_probs: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    计算 DPO Loss

    DPO 的目标：
    - 增加 chosen（好回答）的概率
    - 降低 rejected（差回答）的概率
    - 同时不要偏离参考模型太远

    公式拆解：
    1. π_logratios = log π(chosen) - log π(rejected)
       → 策略模型对好回答和差回答的概率差

    2. ref_logratios = log π_ref(chosen) - log π_ref(rejected)
       → 参考模型对好回答和差回答的概率差

    3. logits = π_logratios - ref_logratios
       → 相对于参考模型的改进

    4. loss = -log σ(β * logits)
       → 希望 logits > 0，即策略模型比参考模型更偏好 chosen

    Args:
        policy_chosen_log_probs: 策略模型对 chosen 的 log prob [batch]
        policy_rejected_log_probs: 策略模型对 rejected 的 log prob [batch]
        ref_chosen_log_probs: 参考模型对 chosen 的 log prob [batch]
        ref_rejected_log_probs: 参考模型对 rejected 的 log prob [batch]
        beta: 温度系数（控制偏离参考模型的程度）

    Returns:
        loss: DPO loss（标量）
    """
    # TODO: 实现 DPO loss
    #
    # Step 1: 计算策略模型的 log ratio
    # pi_logratios = policy_chosen_log_probs - policy_rejected_log_probs
    #
    # Step 2: 计算参考模型的 log ratio
    # ref_logratios = ref_chosen_log_probs - ref_rejected_log_probs
    #
    # Step 3: 计算相对改进
    # logits = pi_logratios - ref_logratios
    #
    # Step 4: 计算 loss
    # loss = -F.logsigmoid(beta * logits).mean()
    #
    # return loss

    raise NotImplementedError("请实现 DPO loss")


# =============================================================================
# 测试代码
# =============================================================================

def test_dpo_loss():
    """测试 DPO loss"""
    print("测试 DPO Loss...")

    batch_size = 4

    # 模拟 log probs
    # 策略模型更偏好 chosen（chosen 概率更高）
    policy_chosen = torch.tensor([-1.0, -1.2, -0.8, -1.1])
    policy_rejected = torch.tensor([-2.0, -2.5, -1.8, -2.2])

    # 参考模型对两者差异不大
    ref_chosen = torch.tensor([-1.5, -1.5, -1.5, -1.5])
    ref_rejected = torch.tensor([-1.6, -1.6, -1.6, -1.6])

    try:
        loss = dpo_loss(
            policy_chosen, policy_rejected,
            ref_chosen, ref_rejected,
            beta=0.1
        )

        assert loss.dim() == 0, "loss 应该是标量"
        assert loss.item() >= 0, "loss 应该非负"

        print(f"✅ DPO Loss 测试通过!")
        print(f"   Loss: {loss.item():.4f}")

        # 解释
        print(f"\n   解释:")
        print(f"   策略模型: chosen 概率更高 (好)")
        print(f"   参考模型: 两者差异不大")
        print(f"   → 策略模型学会了偏好 chosen")

        return True
    except NotImplementedError as e:
        print(f"⚠️ {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("DPO 测试")
    print("=" * 60)
    print()
    test_dpo_loss()
