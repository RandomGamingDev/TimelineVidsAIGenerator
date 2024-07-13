from typing import Union, List

# Helper functions

def log(txt: Union[str, List[str]], loc: str):
	with open(loc, 'w') as f:
		if isinstance(txt, str):
			f.write(txt)
		elif isinstance(txt, list):
			f.writelines(txt)
		else:
			raise TypeError("Only logging of objects that are str | List[str] are allowed")