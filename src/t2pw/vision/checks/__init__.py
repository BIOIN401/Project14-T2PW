"""Development checks for the vision formats (D-W28 c). Not imported by generate() or any production path.

validate     libSBML consistency, SBGN XSD + libSBGN schematron, PWML structure
consistency  cross-format comparison (PWML, SBML, SBGN, KGML parsed back)
faithfulness quotes found in the paper text; database / other-paper elements re-found in cached sources
run_checks   CLI over stored runs

Dev-only libraries (libsbml, lxml) are imported inside functions, never at module import.
"""
