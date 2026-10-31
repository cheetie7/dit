from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch
from torchvision.models import resnet18

from awbc_training.config import ExperimentConfig
from awbc_training.ema import EMA
from awbc_training.inference import load_policy
from test_advantage_weighting import tiny_policy
from test_awbc_data import root


def test_bundled_model_configuration_works_outside_project(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    info = {"features": {"observation.images.top": {"dtype": "video"},
                          "action": {"shape": [8], "dtype": "float32"},
                          "observation.state": {"shape": [8], "dtype": "float32"}}}
    config = ExperimentConfig().policy_configuration(info)
    assert config.dit.action_horizon == 10 and config.dit.action_dim == 8
    assert config.vision.num_cameras == 1


def test_checkpoint_inference_loads_raw_and_ema_without_pretrained_download(tmp_path):
    policy, stats = tiny_policy()
    ema = EMA(policy, 0.9)
    with torch.no_grad():
        ema.policy.dit.velocity_head.bias.fill_(2.)
    config = asdict(policy.config)
    config['vision']['pretrained'] = True
    config['normalization_stats_path'] = '/old/machine/nonexistent/stats.json'
    path = tmp_path/'checkpoint.pt'
    torch.save({'policy': policy.state_dict(), 'ema': ema.state_dict(),
                'policy_config': config, 'normalization_stats': stats}, path)
    with patch('awbc_models.vision.resnet18', wraps=resnet18) as constructor:
        raw = load_policy(path, device='cpu')
        averaged = load_policy(path, use_ema=True, device='cpu')
    assert all(call.kwargs['weights'] is None for call in constructor.call_args_list)
    assert not raw.training and not averaged.training
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(raw.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(averaged.dit.velocity_head.bias, torch.full((3,), 2.))


def test_checkpoint_without_ema_reports_explicit_request(tmp_path):
    policy, stats = tiny_policy()
    path = tmp_path/'checkpoint.pt'
    torch.save({'policy': policy.state_dict(), 'ema': None,
                'policy_config': asdict(policy.config), 'normalization_stats': stats}, path)
    with pytest.raises(ValueError, match='no EMA'):
        load_policy(path, use_ema=True, device='cpu')


@pytest.mark.parametrize('override', [None, 5])
def test_real_cli_uses_file_step_budget_and_cli_override(root, tmp_path, monkeypatch, capsys, override):
    import json
    from awbc_training.train import main
    config = ExperimentConfig()
    config.dataset.root = str(root)
    config.dataset.stratify_key = None
    config.training.num_epochs = 1
    config.training.num_steps = 3
    config.training.device = 'cpu'
    config.training.batch_size = 2
    config.training.num_workers = 0
    config.amp.enabled = config.ema.enabled = config.logging.use_wandb = config.logging.progress_bar = False
    config.scheduler.warmup_steps = 0
    policy, _ = tiny_policy()
    config.policy = asdict(policy.config)
    config.policy['dit'].pop('action_dim')  # Derive the actual action dimension from the dataset.
    config.policy.pop('camera_names')
    config.policy['image_preprocess']['size'] = [32, 48]
    config.policy['normalization_stats_path'] = None
    path = tmp_path/'training.json'
    config.to_json(path)
    output = tmp_path/'cli_run'
    argv = ['awbc-train', '--config', str(path), '--run-dir', str(output)]
    if override is not None:
        argv += ['--set', f'training.num_steps={override}']
    monkeypatch.setattr('sys.argv', argv)
    main()
    expected = override or 3
    state = torch.load(output/'checkpoints/latest_checkpoint.pt', weights_only=False)
    assert state['total_steps'] == state['global_step'] == state['scheduler']['last_epoch'] == expected
    assert state['epoch'] == expected  # one update/epoch: original num_epochs=1 cannot cap it
    assert json.loads((output/'config.json').read_text())['training']['num_steps'] == expected
    assert f'Training budget: steps={expected}; num_epochs is ignored' in capsys.readouterr().out
