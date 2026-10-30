from .core import CLASS, FUNCION

def some_constructor_function(variable: type) -> CLASS:
	return CLASS.method(variable)

__all__ = [
	"CLASS",
	"FUNCTION",
]

# Declaring these allows the user to do:
# from TOOL import CLASS
#
# And import directly from package name, instead of having to know the internal structure: 
# from TOOL.core import CLASS
