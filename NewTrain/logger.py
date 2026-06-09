import logging
import os
from datetime import datetime


class ExperimentLogger:
    """
    实验日志记录器
    为每个实验数据集和运行创建独立的日志文件
    """

    def __init__(self, dataset, run_dir, base_log_dir="logs"):
        """
        初始化日志记录器

        Args:
            dataset: 数据集名称 (Blue, HFUT, PolyU, TJC)
            run_dir: 运行编号 (run_001, run_002, run_003)
            base_log_dir: 日志基础目录
        """
        self.dataset = dataset
        self.run_dir = run_dir
        self.base_log_dir = base_log_dir
        self.experiment_dir = os.path.join(base_log_dir, "experiments", dataset, run_dir)

        # 创建日志目录
        os.makedirs(self.experiment_dir, exist_ok=True)

        # 初始化各个日志记录器
        self.train_logger = self._setup_logger("train", f"{self.experiment_dir}/train.log")
        self.test_logger = self._setup_logger("test", f"{self.experiment_dir}/test.log")
        self.eval_logger = self._setup_logger("eval", f"{self.experiment_dir}/evaluate.log")
        self.main_logger = self._setup_logger("main", f"{self.experiment_dir}/experiment.log")

    def _setup_logger(self, name, log_file):
        """设置日志记录器"""
        logger = logging.getLogger(f"{self.dataset}_{self.run_dir}_{name}")
        logger.setLevel(logging.DEBUG)

        # 避免重复添加handler
        if logger.handlers:
            return logger

        # 文件handler - 记录所有信息
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # 控制台handler - 只显示重要信息
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def log_experiment_start(self, params):
        """记录实验开始信息"""
        self.main_logger.info("=" * 80)
        self.main_logger.info(f"Experiment Started: {self.dataset} - {self.run_dir}")
        self.main_logger.info("=" * 80)
        self.main_logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.main_logger.info("Parameters:")
        for key, value in params.items():
            self.main_logger.info(f"  {key}: {value}")
        self.main_logger.info("=" * 80)

    def log_experiment_end(self, best_result):
        """记录实验结束信息"""
        self.main_logger.info("=" * 80)
        self.main_logger.info(f"Experiment Completed: {self.dataset} - {self.run_dir}")
        self.main_logger.info("=" * 80)
        self.main_logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if best_result:
            self.main_logger.info(f"Best Result: TAR@FAR=1e-6 = {best_result:.6f}")
        self.main_logger.info("=" * 80)

    def log_epoch(self, epoch, loss, accuracy, phase='train'):
        """记录每个epoch的信息"""
        logger = self.train_logger if phase == 'train' else self.test_logger
        logger.info(f"Epoch {epoch}: Loss={loss:.6f}, Accuracy={accuracy:.4f}")

    def log_training_progress(self, epoch, total_epochs, time_elapsed):
        """记录训练进度"""
        self.train_logger.info(f"Progress: {epoch}/{total_epochs} ({epoch/total_epochs*100:.1f}%), Time: {time_elapsed:.2f}s")

    def log_test_result(self, checkpoint_file, score_file_path):
        """记录测试结果"""
        self.test_logger.info(f"Tested checkpoint: {checkpoint_file}")
        self.test_logger.info(f"Score file saved to: {score_file_path}")

    def log_evaluation_result(self, score_file, metrics):
        """记录评估结果"""
        self.eval_logger.info(f"Evaluated score file: {score_file}")
        self.eval_logger.info(f"Metrics: AUC={metrics['AUC']:.6f}, EER={metrics['EER']:.6f}")
        self.eval_logger.info(f"  TAR@FAR=1e-6: {metrics['TAR_FAR_E6']:.6f}")

    def log_error(self, component, error_message):
        """记录错误信息"""
        self.main_logger.error(f"Error in {component}: {error_message}")

    def close(self):
        """关闭所有日志记录器"""
        for logger in [self.train_logger, self.test_logger, self.eval_logger, self.main_logger]:
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)


def get_global_logger(base_log_dir="logs"):
    """
    获取全局日志记录器（用于记录整体实验流程）
    """
    summary_dir = os.path.join(base_log_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    logger = logging.getLogger("global_experiments")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(f"{summary_dir}/experiments.log", mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_error_to_file(component, error_message, base_log_dir="logs"):
    """
    将错误信息记录到专门的错误日志文件
    """
    error_dir = os.path.join(base_log_dir, "errors")
    os.makedirs(error_dir, exist_ok=True)

    error_file = f"{error_dir}/{component}_errors.log"

    with open(error_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Component: {component}\n")
        f.write(f"Error: {error_message}\n")
        f.write(f"{'='*80}\n")
