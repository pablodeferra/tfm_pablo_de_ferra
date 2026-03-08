
\NeedsTeXFormat{LaTeX2e}
\newcommand{\classname}{thesisMDH}
\ProvidesClass{\classname}[2020/05/03 thesisMDH]
\LoadClass[a4paper, titlepage]{article}

\RequirePackage{cancel}
\RequirePackage{colortbl}
\RequirePackage{graphicx}
\graphicspath{{logos/}}
\RequirePackage{mathtools}
\RequirePackage{fancyhdr}
\RequirePackage{enumerate}
\RequirePackage{listings}
\RequirePackage[titletoc]{appendix}
\RequirePackage[margin=3cm]{geometry}
\RequirePackage[absolute]{textpos}
\RequirePackage[section]{placeins}
\RequirePackage{url}
\RequirePackage{tabularx}
\RequirePackage{caption}
\RequirePackage{float}
\RequirePackage{xcolor}
\RequirePackage{titlesec}


\titleformat{\subsection}
  {\normalfont\large\bfseries}{\thesubsection}{1em}{}

\titleformat{\subsubsection}
  {\normalfont}{\thesubsubsection}{1em}{}


\renewcommand{\thesubsubsection}{\arabic{section}.\arabic{subsection}.\arabic{subsubsection}}

% Swedish language package 
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage[english]{babel}


% Bibliography
\usepackage{natbib}
\usepackage[nottoc]{tocbibind}

% \RequirePackage{scrhack} % Loads packages fixes

\NewDocumentCommand{\thesisTitle} { o m }{%
 \IfValueTF{#1}{\def\shorttitle{#1}}{\def\shorttitle{#2}}%
 \def\@title{#2}%
 \def\theTitleName{\huge{\textbf{\uppercase{#2}}}}%
}

\NewDocumentCommand{\authorOne}{m m}{
\newcommand{\authorOneName}{\Large{#1}  \parindent0pt\small{#2}}}

\NewDocumentCommand{\authorTwo}{m m}{
\newcommand{\authorTwoName}{\Large{#1}  \parindent0pt\small{#2}}}

\NewDocumentCommand{\authorThree}{m m}{
\newcommand{\authorThreeName}{\Large{#1}  \parindent0pt\small{#2}}}

\NewDocumentCommand{\subject}{m}{\newcommand{\subjectName}{\Large{#1}}}
\NewDocumentCommand{\university}{m}{\newcommand{\uniName}{\Large{#1}}}
\NewDocumentCommand{\department}{m}{\newcommand{\depName}{\Large{#1}}}
\NewDocumentCommand{\degree}{m}{\newcommand{\degreeName}{\Large{#1}}}


\NewDocumentCommand{\companySupervisors}{m m}{
\newcommand{\compSuperName}{\large{Company Supervisor(s): #1} \\ \small{#2}}}

\NewDocumentCommand{\theDate}{m}{\newcommand{\datum}{\vspace*{\fill}\large #1}}

\newcommand{\warning}[1]{\centering{\textcolor{red}{\large{\textbf{\uppercase{#1\\}}}}}}
\newcommand{\alert}[1]{\textcolor{red}{\textit{\scriptsize\textbf{\uppercase{#1}}}}}


  
\newcommand{\titlePage}{
    \begin{titlepage}
    \begin{center}
\begin{figure}[H]\vspace{-1cm}
    \begin{minipage}{0.5\textwidth}
        \centering
        \includegraphics[width=50mm]{logos/ull.png}
    \end{minipage}
    \begin{minipage}{0.5\textwidth}
        \centering
        \includegraphics[width=50mm]{logos/IAC.png}
    \end{minipage}
\end{figure}

        \vspace{2cm}
    		
            \ifdef{\uniName}{\vspace{-5mm}\uniName\\}{\warning{Missing university name}}
            
            
            \noindent\makebox[\linewidth]{\rule{\textwidth}{0.4pt}} \\
            \vspace{5mm}
            
            \ifdef{\subjectName}{\subjectName}{\warning{Missing course name}} \\
            \vspace{10mm}
            
    		\ifdef{\theTitleName}{\theTitleName}{\warning{Missing title}} \\
    		\vspace{25mm}
    		
    		\ifdef{\authorOneName}{\authorOneName \\ \vspace{5mm}}{\warning{Missing authorOne}} 
    		\Large{Supervisores:} \\ \vspace{0.6cm}
    		\ifdef{\authorTwoName}{\authorTwoName \\ \vspace{5mm}}{}
        \ifdef{\authorThreeName}{\authorThreeName \\ \vspace{5mm}}{}
    		


            \vspace{15mm}

        \end{center}
    \end{titlepage}
}


% Set font
%\setromanfont{Calibri}
\renewcommand\contentsname{Table of Contents}
\newcommand{\HRule}{\rule{\linewidth}{0.5mm}}

\newcommand{\circR}{\textsuperscript{\textregistered}}
% set the style for the section and subsection headers
\renewcommand{\thesection}{\thepart \arabic{section}}
\renewcommand{\thesubsection}{\thepart \arabic{section}.\arabic{subsection}.}

\makeatletter

\newcommand\frontmatter{%
    \cleardoublepage
  %\@mainmatterfalse
  \pagenumbering{roman}}

\newcommand\mainmatter{%
    \cleardoublepage
 % \@mainmattertrue
  \pagenumbering{arabic}}

\newcommand\backmatter{%
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
 % \@mainmatterfalse
  }

\makeatother

\renewenvironment{abstract}{
  \begin{center}%
  \bfseries\abstractname\vspace{-.5em}\vspace{0pt}
  \end{center}
  \itshape
}

% Code style
\lstset{
    backgroundcolor=\color[rgb]{0.92,0.92,0.92},
	basicstyle=\footnotesize,
	showspaces=false,
	showstringspaces=false,
	showtabs=false,
	tabsize=2,
	captionpos=b,
	breaklines=false,
	keywordstyle=\color[rgb]{0,0,1},
	commentstyle=\color[rgb]{0.133,0.545,0.133}
}

% Page style
\pagestyle{fancy}
\marginparsep = 0pt

\newcommand{\fancyHeader}[2]{
\thispagestyle{fancy}
\fancyhead[R]{Pablo de Ferra}
\fancyhead[L]{\leftmark}
% \fancyfoot[L]{}
%\fancyfoot[LE,RO]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
% \renewcommand{\footrulewidth}{0.4pt}
\setlength{\headheight}{12.2pt}

\fancyhfoffset[L]{0.5cm} 
\fancyhfoffset[R]{0.5cm}
}

% Comands
\newcommand{\healpix}{{\sc HEALPix}}
\newcommand{\namaster}{{\sc NaMaster}}
\newcommand{\pysm}{{\sc PySM}}
\newcommand{\commander}{{\sc COMMANDER}}
\newcommand{\python}{{\sc Python}}
\newcommand{\fastcc}{\texttt{fastcc}}
\newcommand{\interpcc}{\texttt{interpcc}}
\newcommand{\IDL}{{\sc IDL}}
\newcommand{\emcee}{{\sc Emcee}}
\newcommand{\sevem}{{\sc SEVEM}}
\endinput

