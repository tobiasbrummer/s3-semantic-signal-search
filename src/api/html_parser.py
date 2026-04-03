#!/usr/bin/env python3
"""
html_to_org.py
Konvertiert Claude/LLM HTML-Exporte in strukturierte .org Dateien.
Code-Blöcke bleiben standardmäßig in der .org Datei (Source Blocks).
Nur explizit benannte Dateien werden nach /code extrahiert.
"""
from bs4 import BeautifulSoup, Tag, Comment, NavigableString
from pathlib import Path
import argparse
import re
import html

# Mapping von Sprachen auf Dateiendungen (für den Fall, dass wir doch extrahieren)
EXT_BY_LANG = {
    "python": "py", "py": "py",
    "javascript": "js", "js": "js", "typescript": "ts", "ts": "ts",
    "bash": "sh", "shell": "sh", "sh": "sh",
    "rust": "rs", "go": "go", "c": "c", "cpp": "cpp",
    "html": "html", "css": "css", "json": "json",
    "yaml": "yaml", "sql": "sql", "markdown": "md", "elisp": "elisp", "emacs-lisp": "elisp"
}

def slugify(s: str) -> str:
    s = s.lower().strip()
    # Ersetze Leerzeichen durch Unterstriche
    s = re.sub(r"\s+", "_", s)
    # Entferne Sonderzeichen außer Wortzeichen und Unterstrichen
    s = re.sub(r"[^\w_]+", "", s)
    return s

class OrgConverter:
    def __init__(self, code_dir: Path, base_slug: str):
        self.code_dir = code_dir
        self.base_slug = base_slug
        self.code_count = 0
        self.code_files = []

    def find_filename(self, node: Tag) -> str:
        """Versucht einen Dateinamen für den Code-Block zu finden."""
        # 1. Suche in data-attributes
        if node.get("data-filename"):
            return node.get("data-filename")
        
        # 2. Suche in Claude-spezifischen Header-Elementen
        prev = node.find_previous_sibling()
        if prev and ("code-block__header" in "".join(prev.get("class", [])) or "filename" in "".join(prev.get("class", [])).lower()):
            return prev.get_text().strip()
        
        # 3. Suche im Code-Inhalt selbst (erste Zeile)
        code_tag = node.find("code") or node
        lines = code_tag.get_text().strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # Erkennt: # filepath: /path/to/file.py, // filename: test.js, /* file: main.c */
            m = re.search(r"(?:filename|filepath|file):\s*([a-zA-Z0-9_\-\./]+)", first_line, re.I)
            if m:
                return Path(m.group(1)).name
            # Erkennt reine Dateinamen in der ersten Zeile (z.B. wave_memory_experiment.py)
            if re.match(r"^[a-zA-Z0-9_\-\.]+\.[a-z]{1,5}$", first_line):
                return first_line
        
        # 4. Suche nach Text wie "File: example.py" direkt vor dem Block
        if prev and prev.name == "div":
            txt = prev.get_text().strip()
            if "." in txt and len(txt) < 50 and not " " in txt:
                return txt
        
        return None

    def is_ascii_art(self, text: str) -> bool:
        """Prüft, ob der Text Rahmenzeichen oder viel ASCII-Art enthält."""
        box_chars = "┌┐└┘├┤┬┴┼─│┃┏┓┗┛┣┫┳┻╋━┃"
        return any(c in text for c in box_chars)

    def is_code_header(self, node: Tag) -> bool:
        """Prüft, ob ein Div nur ein Header für einen Code-Block ist."""
        if node.name != "div":
            return False
        # Nur kleine Divs prüfen, nicht den Haupt-Container
        if len(node.get_text(strip=True)) > 100:
            return False
        if node.find("button"):
            text = node.get_text().strip().lower()
            if text in EXT_BY_LANG or "copy" in text or not text:
                return True
        return False

    def is_download_button(self, btn: Tag) -> bool:
        """Prüft, ob ein Button ein Download-Button ist."""
        text = btn.get_text().strip().lower()
        aria = btn.get("aria-label", "").lower()
        classes = "".join(btn.get("class", [])).lower()
        return any(x in text or x in aria or x in classes for x in ["herunterladen", "download"])

    def find_download_filenames(self, node: Tag) -> list:
        """Sucht nach allen Dateinamen in Download-Containern innerhalb dieses Knotens."""
        # Wir suchen nach Buttons, die wie Download-Buttons aussehen
        buttons = node.find_all("button")
        download_btns = [b for b in buttons if self.is_download_button(b)]
        
        if not download_btns:
            return []

        # Wenn der Knoten zu groß ist (z.B. der ganze Chat), ignorieren wir ihn hier
        # und lassen die Rekursion tiefer zu den spezifischen Containern gehen.
        if len(node.get_text()) > 1000:
            return []

        results = []
        for btn in download_btns:
            # Suche im Umfeld des Buttons nach dem Namen
            # Meistens sind Name, Extension-Label und Button Geschwister
            container = btn.parent.parent
            
            # Wir sammeln alle Text-Teile im Container
            texts = []
            for child in container.descendants:
                if isinstance(child, NavigableString):
                    t = str(child).strip()
                    # Ignoriere UI-Keywords und Icons
                    if t and not self.is_download_button(child.parent) and len(t) > 1:
                        texts.append(t)
            
            if texts:
                # Claude Pattern: ["Wave memory experiment", "PY"]
                if len(texts) >= 2:
                    name = texts[0].lower().replace(" ", "_")
                    if texts[1] == "Dokument":
                        ext_label = texts[2].lower()
                    else:
                        ext_label = texts[1].lower()
                    # Wenn das zweite Element ein kurzes Sprach-Label ist
                    if len(ext_label) <= 4 and any(ext_label == k.lower() for k in EXT_BY_LANG.values()):
                        results.append(f"{name}.{ext_label}")
                    else:
                        results.append(name)
                else:
                    results.append(texts[0])
        return list(dict.fromkeys(results)) # Deduplizieren

    def process_node(self, node, is_root=False):
        if isinstance(node, Comment):
            return ""
        
        if isinstance(node, NavigableString):
            return html.unescape(str(node))

        if not isinstance(node, Tag):
            return ""

        name = node.name.lower()

        # 1. Download-Block Pattern (Claude Artifacts)
        if name == "div" and not is_root:
            node_classes = node.attrs.get("class", [])

            if "group" in node_classes and len(re.findall(r"(?sm)group\s|group\"", str(node))) == 1:
                if "font-user-message" in str(node):
                    print('user')
                    for child in node.descendants:
                        if isinstance(child, Tag) and child.attrs.get("data-testid") == "user-message":
                            question_text = ""

                            for child_text in child.descendants:
                                if isinstance(child_text, NavigableString):
                                    question_text += f"{str(child_text)}\n"

                            question_block = f":QUESTION:\n{question_text}\n:END:\n"
                            return question_block

                elif "font-claude-response" in str(node):
                    print('assistant')


            filenames = self.find_download_filenames(node)
            if filenames:
                links = []
                for fname in filenames:
                    # slug_fname = slugify_filename(fname)
                    # Format: [[file:code/name.py][name.py]]
                    links.append(f"[[file:{self.code_dir.name}/{fname}][{fname}]]")
                # Wir geben die Links zurück und stoppen die Rekursion für diesen Block,
                # damit der UI-Text (wie "PY") nicht doppelt erscheint.
                return "\n" + " ".join(links) + "\n"

        # 2. Filter-Logik für UI/Thinking/Tool-Blöcke
        if not is_root:
            classes = "".join(node.get("class", [])).lower()
            is_ignored_type = (name in ("script", "style", "svg", "button", "nav", "footer", "header", "noscript") or
                "thought" in classes or 
                "thinking" in classes or
                "tool" in classes or
                "call" in classes or
                self.is_code_header(node))
            
            first_child = node.find(recursive=False)
            is_collapse = False
            if first_child and first_child.name == "button":
                child_classes = "".join(first_child.get("class", [])).lower()
                if "collapse-indicator" in child_classes or "group/row" in child_classes:
                    is_collapse = True

            if is_ignored_type or is_collapse:
                # Wir ignorieren den Block, suchen aber nach Dateinamen für Verlinkungen
                pre_blocks = node.find_all("pre")
                links = []
                for pre in pre_blocks:
                    fname = self.find_filename(pre)
                    if fname:
                        # Nur ein Link, kein Extrakt (da manueller Download)
                        links.append(f"\n[[file:{self.code_dir.name}/{fname}][Datei: {fname}]]\n")
                
                return "".join(links)

        # Code-Blöcke (pre) im normalen Textfluss
        if name == "pre":
            code_tag = node.find("code") or node
            code_text = code_tag.get_text().rstrip()
            
            # Sprache erkennen
            lang = "text"
            cls = " ".join(code_tag.get("class", []))
            m = re.search(r"(?:language|lang)-([\w\+\#\-]+)", cls, re.I)
            if m:
                lang = m.group(1).lower()
            
            filename = self.find_filename(node)
            self.code_count += 1

            # Spezialfall: ASCII-Art / Diagramme
            if self.is_ascii_art(code_text) and lang == "text":
                return f"\n\n#+BEGIN_EXAMPLE\n{code_text}\n#+END_EXAMPLE\n\n"

            # Wenn ein Dateiname existiert, extrahieren wir ihn (für normale Chat-Inhalte)
            if filename:
                lines = code_text.split('\n')
                if lines and filename in lines[0]:
                    code_text = '\n'.join(lines[1:]).strip()

                path = self.code_dir / filename
                # Wir schreiben die Datei nur, wenn sie im normalen Chat-Fluss auftaucht
                path.write_text(code_text + "\n", encoding="utf-8")
                self.code_files.append(path)
                return f'\n\n#+INCLUDE: "{path.as_posix()}" src {lang}\n\n'
            
            # Ansonsten: Inline Source Block
            return f"\n\n#+BEGIN_SRC {lang}\n{code_text}\n#+END_SRC\n\n"

        # Überschriften
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            content = self.process_children(node).strip()
            return f"\n\n{'*' * level} {content}\n"

        # Absätze
        if name == "p":
            content = self.process_children(node).strip()
            if not content: return ""
            return f"\n\n{content}\n"

        # Listen
        if name == "ul":
            items = []
            for li in node.find_all("li", recursive=False):
                content = self.process_children(li).strip()
                if content: items.append(f"- {content}")
            return "\n" + "\n".join(items) + "\n"
        
        if name == "ol":
            items = []
            for i, li in enumerate(node.find_all("li", recursive=False), 1):
                content = self.process_children(li).strip()
                if content: items.append(f"{i}. {content}")
            return "\n" + "\n".join(items) + "\n"

        # Inline-Formatierung
        if name == "code":
            # Wenn es innerhalb eines pre ist, wurde es schon behandelt
            if node.parent and node.parent.name == "pre":
                return node.get_text()
            return f"~{node.get_text()}~"
            
        if name in ("b", "strong"):
            return f"*{self.process_children(node)}*"
        if name in ("i", "em"):
            return f"/{self.process_children(node)}/"
        if name == "a":
            href = node.get("href", "")
            content = self.process_children(node)
            return f"[[{href}][{content}]]" if href else content
        
        if name == "br":
            return "\n"

        return self.process_children(node)

    def process_children(self, node):
        return "".join(self.process_node(c) for c in node.children)

def html_to_org(html_path: Path, out_org: Path, code_dir: Path):
    content = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(content, "lxml")
    
    # Claude spezifisch: Suche nach dem Chat-Container
    main_content = soup.find("main") or soup.find("article") or soup.find("body") or soup
    
    base_slug = slugify(html_path.stem)
    converter = OrgConverter(code_dir, base_slug)
    
    # Wir rufen process_node mit is_root=True auf, damit der Haupt-Container nicht gefiltert wird
    org_text = converter.process_node(main_content, is_root=True)
    
    # Bereinigung
    org_text = re.sub(r'\n{3,}', '\n\n', org_text).strip()
    
    if not org_text:
        # Fallback: Wenn alles gefiltert wurde, nimm zumindest den Rohtext
        org_text = f"* {html_path.stem}\n\n" + main_content.get_text(separator="\n\n", strip=True)

    out_org.write_text(org_text + "\n", encoding="utf-8")
    return out_org, converter.code_files

def main():
    parser = argparse.ArgumentParser(description="HTML zu Org-Mode Konverter")
    parser.add_argument("html", type=Path, help="Eingabe HTML")
    parser.add_argument("-o", "--out", type=Path, help="Ausgabe .org")
    parser.add_argument("-c", "--code-dir", type=Path, default=Path("code"), help="Code Ordner")
    args = parser.parse_args()

    html_path = args.html
    out_org = args.out or html_path.with_suffix(".org")
    code_dir = args.code_dir
    
    out_org.parent.mkdir(parents=True, exist_ok=True)
    # Code-Ordner nur erstellen, wenn wir ihn wirklich brauchen
    
    generated_org, code_files = html_to_org(html_path, out_org, code_dir)
    print(f"Wrote: {generated_org}")
    if code_files:
        print(f"Extracted {len(code_files)} files to {code_dir}/")

if __name__ == "__main__":
    main()
