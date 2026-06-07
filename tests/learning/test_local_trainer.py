from titan.learning.local_trainer import LocalTrainer, TrainingConfig


def test_pattern_extraction_naming(tmp_path):
    code_dir = tmp_path / 'src'
    code_dir.mkdir()
    code_file = code_dir / 'app.py'
    code_file.write_text('''
def my_snake_case_function(a, b):
    local_var = a + b
    CONSTANT_VAR = 10
    return local_var + CONSTANT_VAR

class MyPascalClass:
    def __init__(self):
        self._private_member = 5
''')

    trainer = LocalTrainer(config=TrainingConfig(source_dirs=[str(code_dir)], min_file_size=0))
    result = trainer.train(code_dir)

    profile = result.style_profile
    assert profile.function_case == 'snake_case'
    assert profile.class_case == 'PascalCase'
    assert profile.constant_case == 'UPPER_SNAKE_CASE'

def test_pattern_extraction_idioms(tmp_path):
    code_dir = tmp_path / 'src'
    code_dir.mkdir()
    code_file = code_dir / 'idioms.py'
    code_file.write_text('''
data = [x * 2 for x in range(10)]
def typed_func(x: int, y: str) -> str:
    return f'{y}: {x}'
''')

    trainer = LocalTrainer(config=TrainingConfig(min_examples=1, min_file_size=0))
    result = trainer.train(code_dir)

    patterns = result.style_profile.patterns
    pattern_names = [p.name for p in patterns]

    assert 'list_comprehension' in pattern_names
    assert 'type_annotations' in pattern_names

def test_style_adapter_adaptation():
    from titan.learning.local_trainer import StyleAdapter, StyleProfile
    profile = StyleProfile(quote_style='single')
    adapter = StyleAdapter(profile)
    code = 'print("hello world")'
    adapted = adapter.adapt_code(code)
    assert "'hello world'" in adapted

def test_style_adapter_suggestions():
    from titan.learning.local_trainer import StyleAdapter, StyleProfile
    profile = StyleProfile(function_case='camelCase')
    adapter = StyleAdapter(profile)
    bad_code = 'def snake_case_func(): pass'
    suggestions = adapter.suggest_improvements(bad_code)
    assert any('uses snake_case, but project style is camelCase' in s for s in suggestions)
