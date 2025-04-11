# layout when executing code
printwidthleftmargin = 10
printwidth = 100

headerwidth = printwidth + printwidthleftmargin
headersymbol = "x"
########################################################################################################################

author = ": Haoyu Shang "
affiliation = " Peking University "
email = " shy@stu.pku.edu.cn "


def header_message():
    print("\n\n")
    print(headersymbol * headerwidth)
    print(headersymbol)
    print(headersymbol + " nucleon-nucleon scattering solver with chiral Hamiltonian")
    print(headersymbol + " written by " + author)
    print(headersymbol + " Affiliation:" + affiliation)
    print(headersymbol + " email      :" + email)
    print(headersymbol)
    print(headersymbol * headerwidth)


def footer_message():
    section_message("CODE TERMINATED SUCCESFULLY")


def section_message(x):
    print(
        "\n"
        + "*" * (printwidthleftmargin)
        + " "
        + x
        + " "
        + "*" * (printwidth - len(x))
        + "\n"
    )
