<!DOCTYPE html>
<html lang="en">
<head>
        <meta charset="utf-8" />
        <title>First Post on Data Science Blog</title>
        <link rel="stylesheet" href="/theme/css/main.css" />

        <!--[if IE]>
            <script src="https://html5shiv.googlecode.com/svn/trunk/html5.js"></script>
        <![endif]-->
</head>

<body id="index" class="home">
        <header id="banner" class="body">
                <h1><a href="/">umair'blog </a></h1>
                <nav><ul>
                    <li class="active"><a href="/category/posts.html">posts</a></li>
                </ul></nav>
        </header><!-- /#banner -->
<section id="content" class="body">
  <article>
    <header>
      <h1 class="entry-title">
        <a href="/first-post.html" rel="bookmark"
           title="Permalink to First Post on Data Science Blog">First Post on Data Science Blog</a></h1>
    </header>

    <div class="entry-content">
<footer class="post-info">
        <abbr class="published" title="2018-03-02T14:00:00+05:00">
                Published: Fri 02 March 2018
        </abbr>

        <address class="vcard author">
                By                         <a class="url fn" href="/author/umair-qaidr.html">umair qaidr</a>
        </address>
<p>In <a href="/category/posts.html">posts</a>.</p>
<p>tags: <a href="/tag/python-firsts.html">python firsts</a> </p>
</footer><!-- /.post-info -->      <style type="text/css">/*!
*
* IPython notebook
*
*/
/* CSS font colors for translated ANSI colors. */
.ansibold {
  font-weight: bold;
}
/* use dark versions for foreground, to improve visibility */
.ansiblack {
  color: black;
}
.ansired {
  color: darkred;
}
.ansigreen {
  color: darkgreen;
}
.ansiyellow {
  color: #c4a000;
}
.ansiblue {
  color: darkblue;
}
.ansipurple {
  color: darkviolet;
}
.ansicyan {
  color: steelblue;
}
.ansigray {
  color: gray;
}
/* and light for background, for the same reason */
.ansibgblack {
  background-color: black;
}
.ansibgred {
  background-color: red;
}
.ansibggreen {
  background-color: green;
}
.ansibgyellow {
  background-color: yellow;
}
.ansibgblue {
  background-color: blue;
}
.ansibgpurple {
  background-color: magenta;
}
.ansibgcyan {
  background-color: cyan;
}
.ansibggray {
  background-color: gray;
}
div.cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-radius: 2px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  border-width: 1px;
  border-style: solid;
  border-color: transparent;
  width: 100%;
  padding: 5px;
  /* This acts as a spacer between cells, that is outside the border */
  margin: 0px;
  outline: none;
  border-left-width: 1px;
  padding-left: 5px;
  background: linear-gradient(to right, transparent -40px, transparent 1px, transparent 1px, transparent 100%);
}
div.cell.jupyter-soft-selected {
  border-left-color: #90CAF9;
  border-left-color: #E3F2FD;
  border-left-width: 1px;
  padding-left: 5px;
  border-right-color: #E3F2FD;
  border-right-width: 1px;
  background: #E3F2FD;
}
@media print {
  div.cell.jupyter-soft-selected {
    border-color: transparent;
  }
}
div.cell.selected {
  border-color: #ababab;
  border-left-width: 0px;
  padding-left: 6px;
  background: linear-gradient(to right, #42A5F5 -40px, #42A5F5 5px, transparent 5px, transparent 100%);
}
@media print {
  div.cell.selected {
    border-color: transparent;
  }
}
div.cell.selected.jupyter-soft-selected {
  border-left-width: 0;
  padding-left: 6px;
  background: linear-gradient(to right, #42A5F5 -40px, #42A5F5 7px, #E3F2FD 7px, #E3F2FD 100%);
}
.edit_mode div.cell.selected {
  border-color: #66BB6A;
  border-left-width: 0px;
  padding-left: 6px;
  background: linear-gradient(to right, #66BB6A -40px, #66BB6A 5px, transparent 5px, transparent 100%);
}
@media print {
  .edit_mode div.cell.selected {
    border-color: transparent;
  }
}
.prompt {
  /* This needs to be wide enough for 3 digit prompt numbers: In[100]: */
  min-width: 14ex;
  /* This padding is tuned to match the padding on the CodeMirror editor. */
  padding: 0.4em;
  margin: 0px;
  font-family: monospace;
  text-align: right;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
  /* Don't highlight prompt number selection */
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -khtml-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  /* Use default cursor */
  cursor: default;
}
@media (max-width: 540px) {
  .prompt {
    text-align: left;
  }
}
div.inner_cell {
  min-width: 0;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_area {
  border: 1px solid #cfcfcf;
  border-radius: 2px;
  background: #f7f7f7;
  line-height: 1.21429em;
}
/* This is needed so that empty prompt areas can collapse to zero height when there
   is no content in the output_subarea and the prompt. The main purpose of this is
   to make sure that empty JavaScript output_subareas have no height. */
div.prompt:empty {
  padding-top: 0;
  padding-bottom: 0;
}
div.unrecognized_cell {
  padding: 5px 5px 5px 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.unrecognized_cell .inner_cell {
  border-radius: 2px;
  padding: 5px;
  font-weight: bold;
  color: red;
  border: 1px solid #cfcfcf;
  background: #eaeaea;
}
div.unrecognized_cell .inner_cell a {
  color: inherit;
  text-decoration: none;
}
div.unrecognized_cell .inner_cell a:hover {
  color: inherit;
  text-decoration: none;
}
@media (max-width: 540px) {
  div.unrecognized_cell > div.prompt {
    display: none;
  }
}
div.code_cell {
  /* avoid page breaking on code cells when printing */
}
@media print {
  div.code_cell {
    page-break-inside: avoid;
  }
}
/* any special styling for code cells that are currently running goes here */
div.input {
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.input {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_prompt {
  color: #303F9F;
  border-top: 1px solid transparent;
}
div.input_area > div.highlight {
  margin: 0.4em;
  border: none;
  padding: 0px;
  background-color: transparent;
}
div.input_area > div.highlight > pre {
  margin: 0px;
  border: none;
  padding: 0px;
  background-color: transparent;
}
/* The following gets added to the <head> if it is detected that the user has a
 * monospace font with inconsistent normal/bold/italic height.  See
 * notebookmain.js.  Such fonts will have keywords vertically offset with
 * respect to the rest of the text.  The user should select a better font.
 * See: https://github.com/ipython/ipython/issues/1503
 *
 * .CodeMirror span {
 *      vertical-align: bottom;
 * }
 */
.CodeMirror {
  line-height: 1.21429em;
  /* Changed from 1em to our global default */
  font-size: 14px;
  height: auto;
  /* Changed to auto to autogrow */
  background: none;
  /* Changed from white to allow our bg to show through */
}
.CodeMirror-scroll {
  /*  The CodeMirror docs are a bit fuzzy on if overflow-y should be hidden or visible.*/
  /*  We have found that if it is visible, vertical scrollbars appear with font size changes.*/
  overflow-y: hidden;
  overflow-x: auto;
}
.CodeMirror-lines {
  /* In CM2, this used to be 0.4em, but in CM3 it went to 4px. We need the em value because */
  /* we have set a different line-height and want this to scale with that. */
  padding: 0.4em;
}
.CodeMirror-linenumber {
  padding: 0 8px 0 4px;
}
.CodeMirror-gutters {
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.CodeMirror pre {
  /* In CM3 this went to 4px from 0 in CM2. We need the 0 value because of how we size */
  /* .CodeMirror-lines */
  padding: 0;
  border: 0;
  border-radius: 0;
}
/*

Original style from softwaremaniacs.org (c) Ivan Sagalaev <Maniac@SoftwareManiacs.Org>
Adapted from GitHub theme

*/
.highlight-base {
  color: #000;
}
.highlight-variable {
  color: #000;
}
.highlight-variable-2 {
  color: #1a1a1a;
}
.highlight-variable-3 {
  color: #333333;
}
.highlight-string {
  color: #BA2121;
}
.highlight-comment {
  color: #408080;
  font-style: italic;
}
.highlight-number {
  color: #080;
}
.highlight-atom {
  color: #88F;
}
.highlight-keyword {
  color: #008000;
  font-weight: bold;
}
.highlight-builtin {
  color: #008000;
}
.highlight-error {
  color: #f00;
}
.highlight-operator {
  color: #AA22FF;
  font-weight: bold;
}
.highlight-meta {
  color: #AA22FF;
}
/* previously not defined, copying from default codemirror */
.highlight-def {
  color: #00f;
}
.highlight-string-2 {
  color: #f50;
}
.highlight-qualifier {
  color: #555;
}
.highlight-bracket {
  color: #997;
}
.highlight-tag {
  color: #170;
}
.highlight-attribute {
  color: #00c;
}
.highlight-header {
  color: blue;
}
.highlight-quote {
  color: #090;
}
.highlight-link {
  color: #00c;
}
/* apply the same style to codemirror */
.cm-s-ipython span.cm-keyword {
  color: #008000;
  font-weight: bold;
}
.cm-s-ipython span.cm-atom {
  color: #88F;
}
.cm-s-ipython span.cm-number {
  color: #080;
}
.cm-s-ipython span.cm-def {
  color: #00f;
}
.cm-s-ipython span.cm-variable {
  color: #000;
}
.cm-s-ipython span.cm-operator {
  color: #AA22FF;
  font-weight: bold;
}
.cm-s-ipython span.cm-variable-2 {
  color: #1a1a1a;
}
.cm-s-ipython span.cm-variable-3 {
  color: #333333;
}
.cm-s-ipython span.cm-comment {
  color: #408080;
  font-style: italic;
}
.cm-s-ipython span.cm-string {
  color: #BA2121;
}
.cm-s-ipython span.cm-string-2 {
  color: #f50;
}
.cm-s-ipython span.cm-meta {
  color: #AA22FF;
}
.cm-s-ipython span.cm-qualifier {
  color: #555;
}
.cm-s-ipython span.cm-builtin {
  color: #008000;
}
.cm-s-ipython span.cm-bracket {
  color: #997;
}
.cm-s-ipython span.cm-tag {
  color: #170;
}
.cm-s-ipython span.cm-attribute {
  color: #00c;
}
.cm-s-ipython span.cm-header {
  color: blue;
}
.cm-s-ipython span.cm-quote {
  color: #090;
}
.cm-s-ipython span.cm-link {
  color: #00c;
}
.cm-s-ipython span.cm-error {
  color: #f00;
}
.cm-s-ipython span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}
div.output_wrapper {
  /* this position must be relative to enable descendents to be absolute within it */
  position: relative;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  z-index: 1;
}
/* class for the output area when it should be height-limited */
div.output_scroll {
  /* ideally, this would be max-height, but FF barfs all over that */
  height: 24em;
  /* FF needs this *and the wrapper* to specify full width, or it will shrinkwrap */
  width: 100%;
  overflow: auto;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  display: block;
}
/* output div while it is collapsed */
div.output_collapsed {
  margin: 0px;
  padding: 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
div.out_prompt_overlay {
  height: 100%;
  padding: 0px 0.4em;
  position: absolute;
  border-radius: 2px;
}
div.out_prompt_overlay:hover {
  /* use inner shadow to get border that is computed the same on WebKit/FF */
  -webkit-box-shadow: inset 0 0 1px #000;
  box-shadow: inset 0 0 1px #000;
  background: rgba(240, 240, 240, 0.5);
}
div.output_prompt {
  color: #D84315;
}
/* This class is the outer container of all output sections. */
div.output_area {
  padding: 0px;
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.output_area .MathJax_Display {
  text-align: left !important;
}
div.output_area 
div.output_area 
div.output_area img,
div.output_area svg {
  max-width: 100%;
  height: auto;
}
div.output_area img.unconfined,
div.output_area svg.unconfined {
  max-width: none;
}
/* This is needed to protect the pre formating from global settings such
   as that of bootstrap */
.output {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.output_area {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
div.output_area pre {
  margin: 0;
  padding: 0;
  border: 0;
  vertical-align: baseline;
  color: black;
  background-color: transparent;
  border-radius: 0;
}
/* This class is for the output subarea inside the output_area and after
   the prompt div. */
div.output_subarea {
  overflow-x: auto;
  padding: 0.4em;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
  max-width: calc(100% - 14ex);
}
div.output_scroll div.output_subarea {
  overflow-x: visible;
}
/* The rest of the output_* classes are for special styling of the different
   output types */
/* all text output has this class: */
div.output_text {
  text-align: left;
  color: #000;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
}
/* stdout/stderr are 'text' as well as 'stream', but execute_result/error are *not* streams */
div.output_stderr {
  background: #fdd;
  /* very light red background for stderr */
}
div.output_latex {
  text-align: left;
}
/* Empty output_javascript divs should have no height */
div.output_javascript:empty {
  padding: 0;
}
.js-error {
  color: darkred;
}
/* raw_input styles */
div.raw_input_container {
  line-height: 1.21429em;
  padding-top: 5px;
}
pre.raw_input_prompt {
  /* nothing needed here. */
}
input.raw_input {
  font-family: monospace;
  font-size: inherit;
  color: inherit;
  width: auto;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
}
input.raw_input:focus {
  box-shadow: none;
}
p.p-space {
  margin-bottom: 10px;
}
div.output_unrecognized {
  padding: 5px;
  font-weight: bold;
  color: red;
}
div.output_unrecognized a {
  color: inherit;
  text-decoration: none;
}
div.output_unrecognized a:hover {
  color: inherit;
  text-decoration: none;
}
.rendered_html {
  color: #000;
  /* any extras will just be numbers: */
}



.rendered_html :link {
  text-decoration: underline;
}
.rendered_html :visited {
  text-decoration: underline;
}






.rendered_html h1:first-child {
  margin-top: 0.538em;
}
.rendered_html h2:first-child {
  margin-top: 0.636em;
}
.rendered_html h3:first-child {
  margin-top: 0.777em;
}
.rendered_html h4:first-child {
  margin-top: 1em;
}
.rendered_html h5:first-child {
  margin-top: 1em;
}
.rendered_html h6:first-child {
  margin-top: 1em;
}








.rendered_html * + ul {
  margin-top: 1em;
}
.rendered_html * + ol {
  margin-top: 1em;
}


.rendered_html pre,



.rendered_html tr,
.rendered_html th,

.rendered_html td,


.rendered_html * + table {
  margin-top: 1em;
}

.rendered_html * + p {
  margin-top: 1em;
}

.rendered_html * + img {
  margin-top: 1em;
}
.rendered_html img,

.rendered_html img.unconfined,

div.text_cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.text_cell > div.prompt {
    display: none;
  }
}
div.text_cell_render {
  /*font-family: "Helvetica Neue", Arial, Helvetica, Geneva, sans-serif;*/
  outline: none;
  resize: none;
  width: inherit;
  border-style: none;
  padding: 0.5em 0.5em 0.5em 0.4em;
  color: #000;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
a.anchor-link:link {
  text-decoration: none;
  padding: 0px 20px;
  visibility: hidden;
}
h1:hover .anchor-link,
h2:hover .anchor-link,
h3:hover .anchor-link,
h4:hover .anchor-link,
h5:hover .anchor-link,
h6:hover .anchor-link {
  visibility: visible;
}
.text_cell.rendered .input_area {
  display: none;
}
.text_cell.rendered 
.text_cell.unrendered .text_cell_render {
  display: none;
}
.cm-header-1,
.cm-header-2,
.cm-header-3,
.cm-header-4,
.cm-header-5,
.cm-header-6 {
  font-weight: bold;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.cm-header-1 {
  font-size: 185.7%;
}
.cm-header-2 {
  font-size: 157.1%;
}
.cm-header-3 {
  font-size: 128.6%;
}
.cm-header-4 {
  font-size: 110%;
}
.cm-header-5 {
  font-size: 100%;
  font-style: italic;
}
.cm-header-6 {
  font-size: 100%;
  font-style: italic;
}
</style>
<style type="text/css">.highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */</style>
<style type="text/css">
/* Temporary definitions which will become obsolete with Notebook release 5.0 */
.ansi-black-fg { color: #3E424D; }
.ansi-black-bg { background-color: #3E424D; }
.ansi-black-intense-fg { color: #282C36; }
.ansi-black-intense-bg { background-color: #282C36; }
.ansi-red-fg { color: #E75C58; }
.ansi-red-bg { background-color: #E75C58; }
.ansi-red-intense-fg { color: #B22B31; }
.ansi-red-intense-bg { background-color: #B22B31; }
.ansi-green-fg { color: #00A250; }
.ansi-green-bg { background-color: #00A250; }
.ansi-green-intense-fg { color: #007427; }
.ansi-green-intense-bg { background-color: #007427; }
.ansi-yellow-fg { color: #DDB62B; }
.ansi-yellow-bg { background-color: #DDB62B; }
.ansi-yellow-intense-fg { color: #B27D12; }
.ansi-yellow-intense-bg { background-color: #B27D12; }
.ansi-blue-fg { color: #208FFB; }
.ansi-blue-bg { background-color: #208FFB; }
.ansi-blue-intense-fg { color: #0065CA; }
.ansi-blue-intense-bg { background-color: #0065CA; }
.ansi-magenta-fg { color: #D160C4; }
.ansi-magenta-bg { background-color: #D160C4; }
.ansi-magenta-intense-fg { color: #A03196; }
.ansi-magenta-intense-bg { background-color: #A03196; }
.ansi-cyan-fg { color: #60C6C8; }
.ansi-cyan-bg { background-color: #60C6C8; }
.ansi-cyan-intense-fg { color: #258F8F; }
.ansi-cyan-intense-bg { background-color: #258F8F; }
.ansi-white-fg { color: #C5C1B4; }
.ansi-white-bg { background-color: #C5C1B4; }
.ansi-white-intense-fg { color: #A1A6B2; }
.ansi-white-intense-bg { background-color: #A1A6B2; }

.ansi-bold { font-weight: bold; }
</style>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="First-post!">First post!<a class="anchor-link" href="#First-post!">¶</a></h1><p>This will be the first post in our new Pelican blog! You can add any code or markdown cells to this Jupyter notebook, and they will be rendered on your blog.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [2]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">random</span>

<span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">100</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[2]:</div>
<div class="output_text output_subarea output_execute_result">
<pre>4</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [4]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>

<span class="o">%</span><span class="k">matplotlib</span> inline

<span class="n">plt</span><span class="o">.</span><span class="n">hist</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span><span class="mi">5</span><span class="p">,</span><span class="mi">7</span><span class="p">,</span><span class="mi">10</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">,</span><span class="mi">3</span><span class="p">,</span><span class="mi">5</span><span class="p">])</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[4]:</div>
<div class="output_text output_subarea output_execute_result">
<pre>(array([ 2.,  1.,  1.,  0.,  2.,  0.,  1.,  0.,  0.,  2.]),
 array([  1. ,   1.9,   2.8,   3.7,   4.6,   5.5,   6.4,   7.3,   8.2,
          9.1,  10. ]),
 <a list of 10 patch objects>)</pre>
</div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXcAAAEACAYAAABI5zaHAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAADOtJREFUeJzt3H+o3fV9x/HnK4syO1FWoZGZGVer3Sp0WRlWJ8Mz+keN
ZWZ/lNVacLVQpKtoVxhtXSHJf9tgDF0tGmallrpK3agZ/pgTeykKFX9lSjVtpExjOjOGhqKRzR/v
/XGP4e56k3Ny7zn39L73fMCB8+PD97y/ucnzfs/35ntTVUiSelk36wEkSZNn3CWpIeMuSQ0Zd0lq
yLhLUkPGXZIaGhn3JBuTPJDkR0meSnL1EdZdn2Rvkt1JNk9+VEnSuNaPseYN4ItVtTvJicBjSe6r
qj1vL0iyBTizqs5K8mHgRuC86YwsSRpl5JF7Vb1YVbuH918BngFOW7RsK3DrcM3DwMlJNkx4VknS
mI7pnHuSM4DNwMOLXjoN2Lfg8X7e+Q1AkrRKxo778JTMHcA1wyN4SdIvqHHOuZNkPfNh/1ZV3bnE
kv3Ary94vHH43OLt+ItsJGkZqirHsn7cI/dvAE9X1XVHeH0XcDlAkvOAg1V14AgDrsrtpptu4oQT
PgvUFG93cf75Ww6/57Zt21Zt/2Zxm8T+Df8WTPm2vL9nnb9+s9y3X+Sv+Vq5LcfII/ckFwCfAp5K
8sTwT/JaYNP8n2ftrKq7k1yc5FngVeCKZU0jSZqIkXGvqoeAXxpj3VUTmUiStGJeoTpBg8Fg1iNM
lfu3dnXeNy3NuE9Q939A7t/a1XnftDTjLkkNGXdJasi4S1JDxl2SGjLuktSQcZekhoy7JDVk3CWp
IeMuSQ0Zd0lqyLhLUkPGXZIaMu6S1JBxl6SGjLskNWTcJakh4y5JDRl3SWrIuEtSQ8Zdkhoy7pLU
kHGXpIaMuyQ1ZNwlqSHjLkkNGXdJasi4S1JDxl2SGjLuktSQcZekhoy7JDVk3CWpIeMuSQ0Zd0lq
yLhLUkPGXZIaMu6S1JBxl6SGjLskNWTcJakh4y5JDRl3SWrIuEtSQ8Zdkhoy7pLU0Mi4J7k5yYEk
Tx7h9QuTHEzy+PD21cmPKUk6FuvHWHML8HfArUdZ84OqumQyI0mSVmrkkXtVPQi8PGJZJjOOJGkS
JnXO/fwku5PcleQDE9qmJGmZxjktM8pjwOlVdSjJFuB7wNlHWrx9+/bD9weDAYPBYAIjSFIfc3Nz
zM3NrWgbK457Vb2y4P49Sb6e5N1V9dJS6xfGXZL0TosPfHfs2HHM2xj3tEw4wnn1JBsW3D8XyJHC
LklaHSOP3JPcBgyAU5I8D2wDjgeqqnYCH0/yOeB14DXgE9MbV5I0jpFxr6rLRrx+A3DDxCaSJK2Y
V6hKUkPGXZIaMu6S1JBxl6SGjLskNWTcJakh4y5JDRl3SWrIuEtSQ8Zdkhoy7pLUkHGXpIaMuyQ1
ZNwlqSHjLkkNGXdJasi4S1JDxl2SGjLuktSQcZekhoy7JDVk3CWpIeMuSQ0Zd0lqyLhLUkPGXZIa
Mu6S1JBxl6SGjLskNWTcJakh4y5JDRl3SWrIuEtSQ8Zdkhoy7pLUkHGXpIaMuyQ1ZNwlqSHjLkkN
GXdJasi4S1JDxl2SGjLuktSQcZekhoy7JDVk3CWpIeMuSQ2NjHuSm5McSPLkUdZcn2Rvkt1JNk92
REnSsRrnyP0W4KNHejHJFuDMqjoLuBK4cUKzSZKWaWTcq+pB4OWjLNkK3Dpc+zBwcpINkxlPkrQc
kzjnfhqwb8Hj/cPnJEkzsn6133D79u2H7w8GAwaDwWqPMFGPPvoQSab6HuvWvYu33jo01ffYsGET
L77471N9j05OPfUMDhx4bqrv4dfk/6+5uTnm5uZWtI1U1ehFySbgn6vqg0u8diPw/aq6ffh4D3Bh
VR1YYm2N836TsHPnTr7whUd57bWdU3yXu4GPAdPep6zKe6zG12b+G+Ha35cu+7Ea/LNauSRU1TEd
RY57WibD21J2AZcPBzgPOLhU2CVJq2fkaZkktwED4JQkzwPbgOOBqqqdVXV3kouTPAu8ClwxzYEl
SaONjHtVXTbGmqsmM44kaRK8QlWSGjLuktSQcZekhoy7JDVk3CWpIeMuSQ0Zd0lqyLhLUkPGXZIa
Mu6S1JBxl6SGjLskNWTcJakh4y5JDRl3SWrIuEtSQ8Zdkhoy7pLUkHGXpIaMuyQ1ZNwlqSHjLkkN
GXdJasi4S1JDxl2SGjLuktSQcZekhoy7JDVk3CWpIeMuSQ0Zd0lqyLhLUkPGXZIaMu6S1JBxl6SG
jLskNWTcJakh4y5JDRl3SWrIuEtSQ8Zdkhoy7pLUkHGXpIaMuyQ1ZNwlqSHjLkkNGXdJamisuCe5
KMmeJD9J8qUlXr8wycEkjw9vX538qJKkca0ftSDJOuBrwEeAnwGPJLmzqvYsWvqDqrpkCjNKko7R
OEfu5wJ7q+q5qnod+A6wdYl1mehkkqRlGyfupwH7Fjx+YfjcYucn2Z3kriQfmMh0kqRlGXlaZkyP
AadX1aEkW4DvAWcvtXD79u2H7w8GAwaDwYRGkKQe5ubmmJubW9E2xon7fuD0BY83Dp87rKpeWXD/
niRfT/Luqnpp8cYWxl2S9E6LD3x37NhxzNsY57TMI8D7kmxKcjxwKbBr4YIkGxbcPxfIUmGXJK2O
kUfuVfVmkquA+5j/ZnBzVT2T5Mr5l2sn8PEknwNeB14DPjHNoSVJRzfWOfequhd4/6Lnblpw/wbg
hsmOJklaLq9QlaSGjLskNWTcJakh4y5JDRl3SWrIuEtSQ8Zdkhoy7pLUkHGXpIaMuyQ1ZNwlqSHj
LkkNGXdJasi4S1JDxl2SGjLuktSQcZekhoy7JDVk3CWpIeMuSQ0Zd0lqyLhLUkPGXZIaMu6S1JBx
l6SGjLskNWTcJakh4y5JDRl3SWrIuEtSQ8Zdkhoy7pLUkHGXpIaMuyQ1ZNwlqSHjLkkNGXdJasi4
S1JDxl2SGjLuktSQcZekhoy7JDVk3CWpIeMuSQ0Zd0lqyLhLUkNjxT3JRUn2JPlJki8dYc31SfYm
2Z1k82THlCQdi5FxT7IO+BrwUeAc4JNJfnPRmi3AmVV1FnAlcOMUZl0D5mY9wFTNzc3NeoSp6rx/
nfdNSxvnyP1cYG9VPVdVrwPfAbYuWrMVuBWgqh4GTk6yYaKTrglzsx5gqroHovP+dd43LW2cuJ8G
7Fvw+IXhc0dbs3+JNZKkVbJ+1gNMy3HHHUfVvZx00h9O7T3eeOMAhw5NbfOStGypqqMvSM4DtlfV
RcPHXwaqqv5qwZobge9X1e3Dx3uAC6vqwKJtHf3NJElLqqocy/pxjtwfAd6XZBPwH8ClwCcXrdkF
fB64ffjN4ODisC9nOEnS8oyMe1W9meQq4D7mz9HfXFXPJLly/uXaWVV3J7k4ybPAq8AV0x1bknQ0
I0/LSJLWnlW7QnWcC6HWqiQbkzyQ5EdJnkpy9axnmrQk65I8nmTXrGeZtCQnJ/lukmeGX8MPz3qm
SUryleF+PZnk20mOn/VMK5Hk5iQHkjy54LlfTXJfkh8n+ZckJ89yxpU4wv799fDv5+4k/5jkpFHb
WZW4j3Mh1Br3BvDFqjoHOB/4fLP9A7gGeHrWQ0zJdcDdVfVbwG8Dz8x4nokZ/qzss8DvVNUHmT8V
e+lsp1qxW5hvyUJfBu6vqvcDDwBfWfWpJmep/bsPOKeqNgN7GWP/VuvIfZwLodasqnqxqnYP77/C
fBza/D//JBuBi4G/n/UskzY8Avr9qroFoKreqKqfz3isSfo58D/AryRZD7wL+NlsR1qZqnoQeHnR
01uBbw7vfxP4o1UdaoKW2r+qur+q3ho+/CGwcdR2Vivu41wI1UKSM4DNwMOznWSi/hb4c6DjD2h+
A/ivJLcMTzvtTHLCrIealKp6Gfgb4HnmLy48WFX3z3aqqXjP2/9Dr6peBN4z43mm6TPAPaMW+Vsh
JyjJicAdwDXDI/g1L8nHgAPDTyYZ3jpZD3wIuKGqPgQcYv4jfgtJ3gv8GbAJ+DXgxCSXzXaqVdHx
QIQkfwG8XlW3jVq7WnHfD5y+4PHG4XNtDD/y3gF8q6runPU8E3QBcEmSnwL/APxBkltnPNMkvQDs
q6pHh4/vYD72Xfwu8FBVvVRVbwL/BPzejGeahgNv/z6rJKcC/znjeSYuyaeZPz061jfn1Yr74Quh
hj+pv5T5C586+QbwdFVdN+tBJqmqrq2q06vqvcx/3R6oqstnPdekDD/K70ty9vCpj9DrB8c/Bs5L
8stJwvz+dfiB8eJPkbuATw/v/wmw1g+w/s/+JbmI+VOjl1TVf4+zgVX53TJHuhBqNd57NSS5APgU
8FSSJ5j/SHhtVd0728k0pquBbyc5DvgpjS7Cq6p/G37Segx4E3gC2DnbqVYmyW3AADglyfPANuAv
ge8m+QzwHPDHs5twZY6wf9cCxwP/Ov89mh9W1Z8edTtexCRJ/fgDVUlqyLhLUkPGXZIaMu6S1JBx
l6SGjLskNWTcJakh4y5JDf0vZB9FlJ1A1boAAAAASUVORK5CYII=
" />
</div>
</div>
</div>
</div>
</div>

<script type="text/javascript">if (!document.getElementById('mathjaxscript_pelican_#%@#$@#')) {
    var mathjaxscript = document.createElement('script');
    mathjaxscript.id = 'mathjaxscript_pelican_#%@#$@#';
    mathjaxscript.type = 'text/javascript';
    mathjaxscript.src = '//cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML';
    mathjaxscript[(window.opera ? "innerHTML" : "text")] =
        "MathJax.Hub.Config({" +
        "    config: ['MMLorHTML.js']," +
        "    TeX: { extensions: ['AMSmath.js','AMSsymbols.js','noErrors.js','noUndefined.js'], equationNumbers: { autoNumber: 'AMS' } }," +
        "    jax: ['input/TeX','input/MathML','output/HTML-CSS']," +
        "    extensions: ['tex2jax.js','mml2jax.js','MathMenu.js','MathZoom.js']," +
        "    displayAlign: 'center'," +
        "    displayIndent: '0em'," +
        "    showMathMenu: true," +
        "    tex2jax: { " +
        "        inlineMath: [ ['$','$'] ], " +
        "        displayMath: [ ['$$','$$'] ]," +
        "        processEscapes: true," +
        "        preview: 'TeX'," +
        "    }, " +
        "    'HTML-CSS': { " +
        " linebreaks: { automatic: true, width: '95% container' }, " +
        "        styles: { '.MathJax_Display, .MathJax .mo, .MathJax .mi, .MathJax .mn': {color: 'black ! important'} }" +
        "    } " +
        "}); ";
    (document.body || document.getElementsByTagName('head')[0]).appendChild(mathjaxscript);
}
</script>

    </div><!-- /.entry-content -->

  </article>
</section>
        <section id="extras" class="body">
                <div class="blogroll">
                        <h2>blogroll</h2>
                        <ul>
                            <li><a href="http://getpelican.com/">Pelican</a></li>
                            <li><a href="http://python.org/">Python.org</a></li>
                            <li><a href="http://jinja.pocoo.org/">Jinja2</a></li>
                            <li><a href="#">You can modify those links in your config file</a></li>
                        </ul>
                </div><!-- /.blogroll -->
                <div class="social">
                        <h2>social</h2>
                        <ul>

                            <li><a href="#">You can add links in your config file</a></li>
                            <li><a href="#">Another social link</a></li>
                        </ul>
                </div><!-- /.social -->
        </section><!-- /#extras -->

        <footer id="contentinfo" class="body">
                <address id="about" class="vcard body">
                Proudly powered by <a href="http://getpelican.com/">Pelican</a>, which takes great advantage of <a href="http://python.org">Python</a>.
                </address><!-- /#about -->

                <p>The theme is by <a href="http://coding.smashingmagazine.com/2009/08/04/designing-a-html-5-layout-from-scratch/">Smashing Magazine</a>, thanks!</p>
        </footer><!-- /#contentinfo -->

</body>
</html>
