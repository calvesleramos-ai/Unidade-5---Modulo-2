import argparse
import os
import sys

from assistant import HemogramaAssistant


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assistente de hemograma que usa um PDF de referência para responder."    
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Caminho para o arquivo PDF de documentação de referência.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Erro: a variável de ambiente OPENAI_API_KEY não está definida. "
            "Por favor, configure a chave e tente novamente."
        )
        sys.exit(1)

    if not os.path.isfile(args.pdf):
        print(
            f"Erro: não foi possível encontrar o arquivo PDF em '{args.pdf}'. "
            "Verifique o caminho e tente novamente."
        )
        sys.exit(1)

    assistant = HemogramaAssistant(pdf_path=args.pdf, openai_api_key=api_key)

    print("Assistente de hemograma iniciado. Digite 'sair' para encerrar.")
    while True:
        try:
            user_input = input("Informe o hemograma ou sua pergunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o assistente. Até mais!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"sair", "exit", "quit"}:
            print("Encerrando o assistente. Até mais!")
            break

        response = assistant.answer_question(user_input)
        print("\nResposta:\n" + response + "\n")


if __name__ == "__main__":
    main()
