<!DOCTYPE html>

<html lang="en">
<head><meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>Movie Analysis</title><script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<style type="text/css">
    pre { line-height: 125%; }
td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
.highlight .hll { background-color: var(--jp-cell-editor-active-background) }
.highlight { background: var(--jp-cell-editor-background); color: var(--jp-mirror-editor-variable-color) }
.highlight .c { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment */
.highlight .err { color: var(--jp-mirror-editor-error-color) } /* Error */
.highlight .k { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword */
.highlight .o { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator */
.highlight .p { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation */
.highlight .ch { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Multiline */
.highlight .cp { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Preproc */
.highlight .cpf { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Single */
.highlight .cs { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Special */
.highlight .kc { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Pseudo */
.highlight .kr { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Type */
.highlight .m { color: var(--jp-mirror-editor-number-color) } /* Literal.Number */
.highlight .s { color: var(--jp-mirror-editor-string-color) } /* Literal.String */
.highlight .ow { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator.Word */
.highlight .pm { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation.Marker */
.highlight .w { color: var(--jp-mirror-editor-variable-color) } /* Text.Whitespace */
.highlight .mb { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Bin */
.highlight .mf { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Float */
.highlight .mh { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Hex */
.highlight .mi { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer */
.highlight .mo { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Oct */
.highlight .sa { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Affix */
.highlight .sb { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Backtick */
.highlight .sc { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Char */
.highlight .dl { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Delimiter */
.highlight .sd { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Doc */
.highlight .s2 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Double */
.highlight .se { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Escape */
.highlight .sh { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Heredoc */
.highlight .si { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Interpol */
.highlight .sx { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Other */
.highlight .sr { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Regex */
.highlight .s1 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Single */
.highlight .ss { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Symbol */
.highlight .il { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer.Long */
  </style>
<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
 * Mozilla scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */
[data-jp-theme-scrollbars='true'] {
  scrollbar-color: rgb(var(--jp-scrollbar-thumb-color))
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar. These selectors
 * will match lower in the tree, and so will override the above */
[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
}

/* tiny scrollbar */

.jp-scrollbar-tiny {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
  scrollbar-width: thin;
}

/* tiny scrollbar */

.jp-scrollbar-tiny::-webkit-scrollbar,
.jp-scrollbar-tiny::-webkit-scrollbar-corner {
  background-color: transparent;
  height: 4px;
  width: 4px;
}

.jp-scrollbar-tiny::-webkit-scrollbar-thumb {
  background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:horizontal {
  border-left: 0 solid transparent;
  border-right: 0 solid transparent;
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:vertical {
  border-top: 0 solid transparent;
  border-bottom: 0 solid transparent;
}

/*
 * Lumino
 */

.lm-ScrollBar[data-orientation='horizontal'] {
  min-height: 16px;
  max-height: 16px;
  min-width: 45px;
  border-top: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] {
  min-width: 16px;
  max-width: 16px;
  min-height: 45px;
  border-left: 1px solid #a0a0a0;
}

.lm-ScrollBar-button {
  background-color: #f0f0f0;
  background-position: center center;
  min-height: 15px;
  max-height: 15px;
  min-width: 15px;
  max-width: 15px;
}

.lm-ScrollBar-button:hover {
  background-color: #dadada;
}

.lm-ScrollBar-button.lm-mod-active {
  background-color: #cdcdcd;
}

.lm-ScrollBar-track {
  background: #f0f0f0;
}

.lm-ScrollBar-thumb {
  background: #cdcdcd;
}

.lm-ScrollBar-thumb:hover {
  background: #bababa;
}

.lm-ScrollBar-thumb.lm-mod-active {
  background: #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal'] .lm-ScrollBar-thumb {
  height: 100%;
  min-width: 15px;
  border-left: 1px solid #a0a0a0;
  border-right: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] .lm-ScrollBar-thumb {
  width: 100%;
  min-height: 15px;
  border-top: 1px solid #a0a0a0;
  border-bottom: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-left);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-right);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-up);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-down);
  background-size: 17px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-Widget {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
}

.lm-Widget.lm-mod-hidden {
  display: none !important;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.lm-AccordionPanel[data-orientation='horizontal'] > .lm-AccordionPanel-title {
  /* Title is rotated for horizontal accordion panel using CSS */
  display: block;
  transform-origin: top left;
  transform: rotate(-90deg) translate(-100%);
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  display: flex;
  flex-direction: column;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-CommandPalette-search {
  flex: 0 0 auto;
}

.lm-CommandPalette-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  min-height: 0;
  overflow: auto;
  list-style-type: none;
}

.lm-CommandPalette-header {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-CommandPalette-item {
  display: flex;
  flex-direction: row;
}

.lm-CommandPalette-itemIcon {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemContent {
  flex: 1 1 auto;
  overflow: hidden;
}

.lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemLabel {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-close-icon {
  border: 1px solid transparent;
  background-color: transparent;
  position: absolute;
  z-index: 1;
  right: 3%;
  top: 0;
  bottom: 0;
  margin: auto;
  padding: 7px 0;
  display: none;
  vertical-align: middle;
  outline: 0;
  cursor: pointer;
}
.lm-close-icon:after {
  content: 'X';
  display: block;
  width: 15px;
  height: 15px;
  text-align: center;
  color: #000;
  font-weight: normal;
  font-size: 12px;
  cursor: pointer;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-DockPanel {
  z-index: 0;
}

.lm-DockPanel-widget {
  z-index: 0;
}

.lm-DockPanel-tabBar {
  z-index: 1;
}

.lm-DockPanel-handle {
  z-index: 2;
}

.lm-DockPanel-handle.lm-mod-hidden {
  display: none !important;
}

.lm-DockPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}

.lm-DockPanel-handle[data-orientation='horizontal'] {
  cursor: ew-resize;
}

.lm-DockPanel-handle[data-orientation='vertical'] {
  cursor: ns-resize;
}

.lm-DockPanel-handle[data-orientation='horizontal']:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}

.lm-DockPanel-handle[data-orientation='vertical']:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

.lm-DockPanel-overlay {
  z-index: 3;
  box-sizing: border-box;
  pointer-events: none;
}

.lm-DockPanel-overlay.lm-mod-hidden {
  display: none !important;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-Menu {
  z-index: 10000;
  position: absolute;
  white-space: nowrap;
  overflow-x: hidden;
  overflow-y: auto;
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-Menu-content {
  margin: 0;
  padding: 0;
  display: table;
  list-style-type: none;
}

.lm-Menu-item {
  display: table-row;
}

.lm-Menu-item.lm-mod-hidden,
.lm-Menu-item.lm-mod-collapsed {
  display: none !important;
}

.lm-Menu-itemIcon,
.lm-Menu-itemSubmenuIcon {
  display: table-cell;
  text-align: center;
}

.lm-Menu-itemLabel {
  display: table-cell;
  text-align: left;
}

.lm-Menu-itemShortcut {
  display: table-cell;
  text-align: right;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-MenuBar {
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-MenuBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: row;
  list-style-type: none;
}

.lm-MenuBar-item {
  box-sizing: border-box;
}

.lm-MenuBar-itemIcon,
.lm-MenuBar-itemLabel {
  display: inline-block;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-ScrollBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-ScrollBar[data-orientation='horizontal'] {
  flex-direction: row;
}

.lm-ScrollBar[data-orientation='vertical'] {
  flex-direction: column;
}

.lm-ScrollBar-button {
  box-sizing: border-box;
  flex: 0 0 auto;
}

.lm-ScrollBar-track {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  flex: 1 1 auto;
}

.lm-ScrollBar-thumb {
  box-sizing: border-box;
  position: absolute;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-SplitPanel-child {
  z-index: 0;
}

.lm-SplitPanel-handle {
  z-index: 1;
}

.lm-SplitPanel-handle.lm-mod-hidden {
  display: none !important;
}

.lm-SplitPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}

.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle {
  cursor: ew-resize;
}

.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle {
  cursor: ns-resize;
}

.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}

.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-TabBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-TabBar[data-orientation='horizontal'] {
  flex-direction: row;
  align-items: flex-end;
}

.lm-TabBar[data-orientation='vertical'] {
  flex-direction: column;
  align-items: flex-end;
}

.lm-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex: 1 1 auto;
  list-style-type: none;
}

.lm-TabBar[data-orientation='horizontal'] > .lm-TabBar-content {
  flex-direction: row;
}

.lm-TabBar[data-orientation='vertical'] > .lm-TabBar-content {
  flex-direction: column;
}

.lm-TabBar-tab {
  display: flex;
  flex-direction: row;
  box-sizing: border-box;
  overflow: hidden;
  touch-action: none; /* Disable native Drag/Drop */
}

.lm-TabBar-tabIcon,
.lm-TabBar-tabCloseIcon {
  flex: 0 0 auto;
}

.lm-TabBar-tabLabel {
  flex: 1 1 auto;
  overflow: hidden;
  white-space: nowrap;
}

.lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing: border-box;
}

.lm-TabBar-tab.lm-mod-hidden {
  display: none !important;
}

.lm-TabBar-addButton.lm-mod-hidden {
  display: none !important;
}

.lm-TabBar.lm-mod-dragging .lm-TabBar-tab {
  position: relative;
}

.lm-TabBar.lm-mod-dragging[data-orientation='horizontal'] .lm-TabBar-tab {
  left: 0;
  transition: left 150ms ease;
}

.lm-TabBar.lm-mod-dragging[data-orientation='vertical'] .lm-TabBar-tab {
  top: 0;
  transition: top 150ms ease;
}

.lm-TabBar.lm-mod-dragging .lm-TabBar-tab.lm-mod-dragging {
  transition: none;
}

.lm-TabBar-tabLabel .lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing: border-box;
  background: inherit;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-TabPanel-tabBar {
  z-index: 1;
}

.lm-TabPanel-stackedPanel {
  z-index: 0;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapse {
  display: flex;
  flex-direction: column;
  align-items: stretch;
}

.jp-Collapse-header {
  padding: 1px 12px;
  background-color: var(--jp-layout-color1);
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  align-items: center;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  text-transform: uppercase;
  user-select: none;
}

.jp-Collapser-icon {
  height: 16px;
}

.jp-Collapse-header-collapsed .jp-Collapser-icon {
  transform: rotate(-90deg);
  margin: auto 0;
}

.jp-Collapser-title {
  line-height: 25px;
}

.jp-Collapse-contents {
  padding: 0 12px;
  background-color: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensureUiComponents() in @jupyterlab/buildutils */

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

/* Icons urls */

:root {
  --jp-icon-add-above: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzEzN18xOTQ5MikiPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGQ9Ik00Ljc1IDQuOTMwNjZINi42MjVWNi44MDU2NkM2LjYyNSA3LjAxMTkxIDYuNzkzNzUgNy4xODA2NiA3IDcuMTgwNjZDNy4yMDYyNSA3LjE4MDY2IDcuMzc1IDcuMDExOTEgNy4zNzUgNi44MDU2NlY0LjkzMDY2SDkuMjVDOS40NTYyNSA0LjkzMDY2IDkuNjI1IDQuNzYxOTEgOS42MjUgNC41NTU2NkM5LjYyNSA0LjM0OTQxIDkuNDU2MjUgNC4xODA2NiA5LjI1IDQuMTgwNjZINy4zNzVWMi4zMDU2NkM3LjM3NSAyLjA5OTQxIDcuMjA2MjUgMS45MzA2NiA3IDEuOTMwNjZDNi43OTM3NSAxLjkzMDY2IDYuNjI1IDIuMDk5NDEgNi42MjUgMi4zMDU2NlY0LjE4MDY2SDQuNzVDNC41NDM3NSA0LjE4MDY2IDQuMzc1IDQuMzQ5NDEgNC4zNzUgNC41NTU2NkM0LjM3NSA0Ljc2MTkxIDQuNTQzNzUgNC45MzA2NiA0Ljc1IDQuOTMwNjZaIiBmaWxsPSIjNjE2MTYxIiBzdHJva2U9IiM2MTYxNjEiIHN0cm9rZS13aWR0aD0iMC43Ii8+CjwvZz4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTExLjUgOS41VjExLjVMMi41IDExLjVWOS41TDExLjUgOS41Wk0xMiA4QzEyLjU1MjMgOCAxMyA4LjQ0NzcyIDEzIDlWMTJDMTMgMTIuNTUyMyAxMi41NTIzIDEzIDEyIDEzTDIgMTNDMS40NDc3MiAxMyAxIDEyLjU1MjMgMSAxMlY5QzEgOC40NDc3MiAxLjQ0NzcxIDggMiA4TDEyIDhaIiBmaWxsPSIjNjE2MTYxIi8+CjxkZWZzPgo8Y2xpcFBhdGggaWQ9ImNsaXAwXzEzN18xOTQ5MiI+CjxyZWN0IGNsYXNzPSJqcC1pY29uMyIgd2lkdGg9IjYiIGhlaWdodD0iNiIgZmlsbD0id2hpdGUiIHRyYW5zZm9ybT0ibWF0cml4KC0xIDAgMCAxIDEwIDEuNTU1NjYpIi8+CjwvY2xpcFBhdGg+CjwvZGVmcz4KPC9zdmc+Cg==);
  --jp-icon-add-below: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzEzN18xOTQ5OCkiPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGQ9Ik05LjI1IDEwLjA2OTNMNy4zNzUgMTAuMDY5M0w3LjM3NSA4LjE5NDM0QzcuMzc1IDcuOTg4MDkgNy4yMDYyNSA3LjgxOTM0IDcgNy44MTkzNEM2Ljc5Mzc1IDcuODE5MzQgNi42MjUgNy45ODgwOSA2LjYyNSA4LjE5NDM0TDYuNjI1IDEwLjA2OTNMNC43NSAxMC4wNjkzQzQuNTQzNzUgMTAuMDY5MyA0LjM3NSAxMC4yMzgxIDQuMzc1IDEwLjQ0NDNDNC4zNzUgMTAuNjUwNiA0LjU0Mzc1IDEwLjgxOTMgNC43NSAxMC44MTkzTDYuNjI1IDEwLjgxOTNMNi42MjUgMTIuNjk0M0M2LjYyNSAxMi45MDA2IDYuNzkzNzUgMTMuMDY5MyA3IDEzLjA2OTNDNy4yMDYyNSAxMy4wNjkzIDcuMzc1IDEyLjkwMDYgNy4zNzUgMTIuNjk0M0w3LjM3NSAxMC44MTkzTDkuMjUgMTAuODE5M0M5LjQ1NjI1IDEwLjgxOTMgOS42MjUgMTAuNjUwNiA5LjYyNSAxMC40NDQzQzkuNjI1IDEwLjIzODEgOS40NTYyNSAxMC4wNjkzIDkuMjUgMTAuMDY5M1oiIGZpbGw9IiM2MTYxNjEiIHN0cm9rZT0iIzYxNjE2MSIgc3Ryb2tlLXdpZHRoPSIwLjciLz4KPC9nPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMi41IDUuNUwyLjUgMy41TDExLjUgMy41TDExLjUgNS41TDIuNSA1LjVaTTIgN0MxLjQ0NzcyIDcgMSA2LjU1MjI4IDEgNkwxIDNDMSAyLjQ0NzcyIDEuNDQ3NzIgMiAyIDJMMTIgMkMxMi41NTIzIDIgMTMgMi40NDc3MiAxMyAzTDEzIDZDMTMgNi41NTIyOSAxMi41NTIzIDcgMTIgN0wyIDdaIiBmaWxsPSIjNjE2MTYxIi8+CjxkZWZzPgo8Y2xpcFBhdGggaWQ9ImNsaXAwXzEzN18xOTQ5OCI+CjxyZWN0IGNsYXNzPSJqcC1pY29uMyIgd2lkdGg9IjYiIGhlaWdodD0iNiIgZmlsbD0id2hpdGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMS43NDg0NmUtMDcgMS43NDg0NmUtMDcgLTEgNCAxMy40NDQzKSIvPgo8L2NsaXBQYXRoPgo8L2RlZnM+Cjwvc3ZnPgo=);
  --jp-icon-add: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDEzaC02djZoLTJ2LTZINXYtMmg2VjVoMnY2aDZ2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bell: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE2IDE2IiB2ZXJzaW9uPSIxLjEiPgogICA8cGF0aCBjbGFzcz0ianAtaWNvbjIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMzMzMzMzIgogICAgICBkPSJtOCAwLjI5Yy0xLjQgMC0yLjcgMC43My0zLjYgMS44LTEuMiAxLjUtMS40IDMuNC0xLjUgNS4yLTAuMTggMi4yLTAuNDQgNC0yLjMgNS4zbDAuMjggMS4zaDVjMC4wMjYgMC42NiAwLjMyIDEuMSAwLjcxIDEuNSAwLjg0IDAuNjEgMiAwLjYxIDIuOCAwIDAuNTItMC40IDAuNi0xIDAuNzEtMS41aDVsMC4yOC0xLjNjLTEuOS0wLjk3LTIuMi0zLjMtMi4zLTUuMy0wLjEzLTEuOC0wLjI2LTMuNy0xLjUtNS4yLTAuODUtMS0yLjItMS44LTMuNi0xLjh6bTAgMS40YzAuODggMCAxLjkgMC41NSAyLjUgMS4zIDAuODggMS4xIDEuMSAyLjcgMS4yIDQuNCAwLjEzIDEuNyAwLjIzIDMuNiAxLjMgNS4yaC0xMGMxLjEtMS42IDEuMi0zLjQgMS4zLTUuMiAwLjEzLTEuNyAwLjMtMy4zIDEuMi00LjQgMC41OS0wLjcyIDEuNi0xLjMgMi41LTEuM3ptLTAuNzQgMTJoMS41Yy0wLjAwMTUgMC4yOCAwLjAxNSAwLjc5LTAuNzQgMC43OS0wLjczIDAuMDAxNi0wLjcyLTAuNTMtMC43NC0wLjc5eiIgLz4KPC9zdmc+Cg==);
  --jp-icon-bug-dot: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiPgogICAgICAgIDxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMTcuMTkgOEgyMFYxMEgxNy45MUMxNy45NiAxMC4zMyAxOCAxMC42NiAxOCAxMVYxMkgyMFYxNEgxOC41SDE4VjE0LjAyNzVDMTUuNzUgMTQuMjc2MiAxNCAxNi4xODM3IDE0IDE4LjVDMTQgMTkuMjA4IDE0LjE2MzUgMTkuODc3OSAxNC40NTQ5IDIwLjQ3MzlDMTMuNzA2MyAyMC44MTE3IDEyLjg3NTcgMjEgMTIgMjFDOS43OCAyMSA3Ljg1IDE5Ljc5IDYuODEgMThINFYxNkg2LjA5QzYuMDQgMTUuNjcgNiAxNS4zNCA2IDE1VjE0SDRWMTJINlYxMUM2IDEwLjY2IDYuMDQgMTAuMzMgNi4wOSAxMEg0VjhINi44MUM3LjI2IDcuMjIgNy44OCA2LjU1IDguNjIgNi4wNEw3IDQuNDFMOC40MSAzTDEwLjU5IDUuMTdDMTEuMDQgNS4wNiAxMS41MSA1IDEyIDVDMTIuNDkgNSAxMi45NiA1LjA2IDEzLjQyIDUuMTdMMTUuNTkgM0wxNyA0LjQxTDE1LjM3IDYuMDRDMTYuMTIgNi41NSAxNi43NCA3LjIyIDE3LjE5IDhaTTEwIDE2SDE0VjE0SDEwVjE2Wk0xMCAxMkgxNFYxMEgxMFYxMloiIGZpbGw9IiM2MTYxNjEiLz4KICAgICAgICA8cGF0aCBkPSJNMjIgMTguNUMyMiAyMC40MzMgMjAuNDMzIDIyIDE4LjUgMjJDMTYuNTY3IDIyIDE1IDIwLjQzMyAxNSAxOC41QzE1IDE2LjU2NyAxNi41NjcgMTUgMTguNSAxNUMyMC40MzMgMTUgMjIgMTYuNTY3IDIyIDE4LjVaIiBmaWxsPSIjNjE2MTYxIi8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bug: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0yMCA4aC0yLjgxYy0uNDUtLjc4LTEuMDctMS40NS0xLjgyLTEuOTZMMTcgNC40MSAxNS41OSAzbC0yLjE3IDIuMTdDMTIuOTYgNS4wNiAxMi40OSA1IDEyIDVjLS40OSAwLS45Ni4wNi0xLjQxLjE3TDguNDEgMyA3IDQuNDFsMS42MiAxLjYzQzcuODggNi41NSA3LjI2IDcuMjIgNi44MSA4SDR2MmgyLjA5Yy0uMDUuMzMtLjA5LjY2LS4wOSAxdjFINHYyaDJ2MWMwIC4zNC4wNC42Ny4wOSAxSDR2MmgyLjgxYzEuMDQgMS43OSAyLjk3IDMgNS4xOSAzczQuMTUtMS4yMSA1LjE5LTNIMjB2LTJoLTIuMDljLjA1LS4zMy4wOS0uNjYuMDktMXYtMWgydi0yaC0ydi0xYzAtLjM0LS4wNC0uNjctLjA5LTFIMjBWOHptLTYgOGgtNHYtMmg0djJ6bTAtNGgtNHYtMmg0djJ6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-build: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE0LjkgMTcuNDVDMTYuMjUgMTcuNDUgMTcuMzUgMTYuMzUgMTcuMzUgMTVDMTcuMzUgMTMuNjUgMTYuMjUgMTIuNTUgMTQuOSAxMi41NUMxMy41NCAxMi41NSAxMi40NSAxMy42NSAxMi40NSAxNUMxMi40NSAxNi4zNSAxMy41NCAxNy40NSAxNC45IDE3LjQ1Wk0yMC4xIDE1LjY4TDIxLjU4IDE2Ljg0QzIxLjcxIDE2Ljk1IDIxLjc1IDE3LjEzIDIxLjY2IDE3LjI5TDIwLjI2IDE5LjcxQzIwLjE3IDE5Ljg2IDIwIDE5LjkyIDE5LjgzIDE5Ljg2TDE4LjA5IDE5LjE2QzE3LjczIDE5LjQ0IDE3LjMzIDE5LjY3IDE2LjkxIDE5Ljg1TDE2LjY0IDIxLjdDMTYuNjIgMjEuODcgMTYuNDcgMjIgMTYuMyAyMkgxMy41QzEzLjMyIDIyIDEzLjE4IDIxLjg3IDEzLjE1IDIxLjdMMTIuODkgMTkuODVDMTIuNDYgMTkuNjcgMTIuMDcgMTkuNDQgMTEuNzEgMTkuMTZMOS45NjAwMiAxOS44NkM5LjgxMDAyIDE5LjkyIDkuNjIwMDIgMTkuODYgOS41NDAwMiAxOS43MUw4LjE0MDAyIDE3LjI5QzguMDUwMDIgMTcuMTMgOC4wOTAwMiAxNi45NSA4LjIyMDAyIDE2Ljg0TDkuNzAwMDIgMTUuNjhMOS42NTAwMSAxNUw5LjcwMDAyIDE0LjMxTDguMjIwMDIgMTMuMTZDOC4wOTAwMiAxMy4wNSA4LjA1MDAyIDEyLjg2IDguMTQwMDIgMTIuNzFMOS41NDAwMiAxMC4yOUM5LjYyMDAyIDEwLjEzIDkuODEwMDIgMTAuMDcgOS45NjAwMiAxMC4xM0wxMS43MSAxMC44NEMxMi4wNyAxMC41NiAxMi40NiAxMC4zMiAxMi44OSAxMC4xNUwxMy4xNSA4LjI4OTk4QzEzLjE4IDguMTI5OTggMTMuMzIgNy45OTk5OCAxMy41IDcuOTk5OThIMTYuM0MxNi40NyA3Ljk5OTk4IDE2LjYyIDguMTI5OTggMTYuNjQgOC4yODk5OEwxNi45MSAxMC4xNUMxNy4zMyAxMC4zMiAxNy43MyAxMC41NiAxOC4wOSAxMC44NEwxOS44MyAxMC4xM0MyMCAxMC4wNyAyMC4xNyAxMC4xMyAyMC4yNiAxMC4yOUwyMS42NiAxMi43MUMyMS43NSAxMi44NiAyMS43MSAxMy4wNSAyMS41OCAxMy4xNkwyMC4xIDE0LjMxTDIwLjE1IDE1TDIwLjEgMTUuNjhaIi8+CiAgICA8cGF0aCBkPSJNNy4zMjk2NiA3LjQ0NDU0QzguMDgzMSA3LjAwOTU0IDguMzM5MzIgNi4wNTMzMiA3LjkwNDMyIDUuMjk5ODhDNy40NjkzMiA0LjU0NjQzIDYuNTA4MSA0LjI4MTU2IDUuNzU0NjYgNC43MTY1NkM1LjM5MTc2IDQuOTI2MDggNS4xMjY5NSA1LjI3MTE4IDUuMDE4NDkgNS42NzU5NEM0LjkxMDA0IDYuMDgwNzEgNC45NjY4MiA2LjUxMTk4IDUuMTc2MzQgNi44NzQ4OEM1LjYxMTM0IDcuNjI4MzIgNi41NzYyMiA3Ljg3OTU0IDcuMzI5NjYgNy40NDQ1NFpNOS42NTcxOCA0Ljc5NTkzTDEwLjg2NzIgNC45NTE3OUMxMC45NjI4IDQuOTc3NDEgMTEuMDQwMiA1LjA3MTMzIDExLjAzODIgNS4xODc5M0wxMS4wMzg4IDYuOTg4OTNDMTEuMDQ1NSA3LjEwMDU0IDEwLjk2MTYgNy4xOTUxOCAxMC44NTUgNy4yMTA1NEw5LjY2MDAxIDcuMzgwODNMOS4yMzkxNSA4LjEzMTg4TDkuNjY5NjEgOS4yNTc0NUM5LjcwNzI5IDkuMzYyNzEgOS42NjkzNCA5LjQ3Njk5IDkuNTc0MDggOS41MzE5OUw4LjAxNTIzIDEwLjQzMkM3LjkxMTMxIDEwLjQ5MiA3Ljc5MzM3IDEwLjQ2NzcgNy43MjEwNSAxMC4zODI0TDYuOTg3NDggOS40MzE4OEw2LjEwOTMxIDkuNDMwODNMNS4zNDcwNCAxMC4zOTA1QzUuMjg5MDkgMTAuNDcwMiA1LjE3MzgzIDEwLjQ5MDUgNS4wNzE4NyAxMC40MzM5TDMuNTEyNDUgOS41MzI5M0MzLjQxMDQ5IDkuNDc2MzMgMy4zNzY0NyA5LjM1NzQxIDMuNDEwNzUgOS4yNTY3OUwzLjg2MzQ3IDguMTQwOTNMMy42MTc0OSA3Ljc3NDg4TDMuNDIzNDcgNy4zNzg4M0wyLjIzMDc1IDcuMjEyOTdDMi4xMjY0NyA3LjE5MjM1IDIuMDQwNDkgNy4xMDM0MiAyLjA0MjQ1IDYuOTg2ODJMMi4wNDE4NyA1LjE4NTgyQzIuMDQzODMgNS4wNjkyMiAyLjExOTA5IDQuOTc5NTggMi4yMTcwNCA0Ljk2OTIyTDMuNDIwNjUgNC43OTM5M0wzLjg2NzQ5IDQuMDI3ODhMMy40MTEwNSAyLjkxNzMxQzMuMzczMzcgMi44MTIwNCAzLjQxMTMxIDIuNjk3NzYgMy41MTUyMyAyLjYzNzc2TDUuMDc0MDggMS43Mzc3NkM1LjE2OTM0IDEuNjgyNzYgNS4yODcyOSAxLjcwNzA0IDUuMzU5NjEgMS43OTIzMUw2LjExOTE1IDIuNzI3ODhMNi45ODAwMSAyLjczODkzTDcuNzI0OTYgMS43ODkyMkM3Ljc5MTU2IDEuNzA0NTggNy45MTU0OCAxLjY3OTIyIDguMDA4NzkgMS43NDA4Mkw5LjU2ODIxIDIuNjQxODJDOS42NzAxNyAyLjY5ODQyIDkuNzEyODUgMi44MTIzNCA5LjY4NzIzIDIuOTA3OTdMOS4yMTcxOCA0LjAzMzgzTDkuNDYzMTYgNC4zOTk4OEw5LjY1NzE4IDQuNzk1OTNaIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iOS45LDEzLjYgMy42LDcuNCA0LjQsNi42IDkuOSwxMi4yIDE1LjQsNi43IDE2LjEsNy40ICIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNS45TDksOS43bDMuOC0zLjhsMS4yLDEuMmwtNC45LDVsLTQuOS01TDUuMiw1Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNy41TDksMTEuMmwzLjgtMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-left: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik0xMC44LDEyLjhMNy4xLDlsMy44LTMuOGwwLDcuNkgxMC44eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-right: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik03LjIsNS4yTDEwLjksOWwtMy44LDMuOFY1LjJINy4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-up-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iMTUuNCwxMy4zIDkuOSw3LjcgNC40LDEzLjIgMy42LDEyLjUgOS45LDYuMyAxNi4xLDEyLjYgIi8+Cgk8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-up: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik01LjIsMTAuNUw5LDYuOGwzLjgsMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-case-sensitive: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWFjY2VudDIiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTcuNiw4aDAuOWwzLjUsOGgtMS4xTDEwLDE0SDZsLTAuOSwySDRMNy42LDh6IE04LDkuMUw2LjQsMTNoMy4yTDgsOS4xeiIvPgogICAgPHBhdGggZD0iTTE2LjYsOS44Yy0wLjIsMC4xLTAuNCwwLjEtMC43LDAuMWMtMC4yLDAtMC40LTAuMS0wLjYtMC4yYy0wLjEtMC4xLTAuMi0wLjQtMC4yLTAuNyBjLTAuMywwLjMtMC42LDAuNS0wLjksMC43Yy0wLjMsMC4xLTAuNywwLjItMS4xLDAuMmMtMC4zLDAtMC41LDAtMC43LTAuMWMtMC4yLTAuMS0wLjQtMC4yLTAuNi0wLjNjLTAuMi0wLjEtMC4zLTAuMy0wLjQtMC41IGMtMC4xLTAuMi0wLjEtMC40LTAuMS0wLjdjMC0wLjMsMC4xLTAuNiwwLjItMC44YzAuMS0wLjIsMC4zLTAuNCwwLjQtMC41QzEyLDcsMTIuMiw2LjksMTIuNSw2LjhjMC4yLTAuMSwwLjUtMC4xLDAuNy0wLjIgYzAuMy0wLjEsMC41LTAuMSwwLjctMC4xYzAuMiwwLDAuNC0wLjEsMC42LTAuMWMwLjIsMCwwLjMtMC4xLDAuNC0wLjJjMC4xLTAuMSwwLjItMC4yLDAuMi0wLjRjMC0xLTEuMS0xLTEuMy0xIGMtMC40LDAtMS40LDAtMS40LDEuMmgtMC45YzAtMC40LDAuMS0wLjcsMC4yLTFjMC4xLTAuMiwwLjMtMC40LDAuNS0wLjZjMC4yLTAuMiwwLjUtMC4zLDAuOC0wLjNDMTMuMyw0LDEzLjYsNCwxMy45LDQgYzAuMywwLDAuNSwwLDAuOCwwLjFjMC4zLDAsMC41LDAuMSwwLjcsMC4yYzAuMiwwLjEsMC40LDAuMywwLjUsMC41QzE2LDUsMTYsNS4yLDE2LDUuNnYyLjljMCwwLjIsMCwwLjQsMCwwLjUgYzAsMC4xLDAuMSwwLjIsMC4zLDAuMmMwLjEsMCwwLjIsMCwwLjMsMFY5Ljh6IE0xNS4yLDYuOWMtMS4yLDAuNi0zLjEsMC4yLTMuMSwxLjRjMCwxLjQsMy4xLDEsMy4xLTAuNVY2Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik05IDE2LjE3TDQuODMgMTJsLTEuNDIgMS40MUw5IDE5IDIxIDdsLTEuNDEtMS40MXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-circle-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDJDNi40NyAyIDIgNi40NyAyIDEyczQuNDcgMTAgMTAgMTAgMTAtNC40NyAxMC0xMFMxNy41MyAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-circle: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iOSIgY3k9IjkiIHI9IjgiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-clear: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8bWFzayBpZD0iZG9udXRIb2xlIj4KICAgIDxyZWN0IHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0id2hpdGUiIC8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI4IiBmaWxsPSJibGFjayIvPgogIDwvbWFzaz4KCiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxyZWN0IGhlaWdodD0iMTgiIHdpZHRoPSIyIiB4PSIxMSIgeT0iMyIgdHJhbnNmb3JtPSJyb3RhdGUoMzE1LCAxMiwgMTIpIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgbWFzaz0idXJsKCNkb251dEhvbGUpIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-close: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1ub25lIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIGpwLWljb24zLWhvdmVyIiBmaWxsPSJub25lIj4KICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjExIi8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIGpwLWljb24tYWNjZW50Mi1ob3ZlciIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMTkgNi40MUwxNy41OSA1IDEyIDEwLjU5IDYuNDEgNSA1IDYuNDEgMTAuNTkgMTIgNSAxNy41OSA2LjQxIDE5IDEyIDEzLjQxIDE3LjU5IDE5IDE5IDE3LjU5IDEzLjQxIDEyeiIvPgogIDwvZz4KCiAgPGcgY2xhc3M9ImpwLWljb24tbm9uZSBqcC1pY29uLWJ1c3kiIGZpbGw9Im5vbmUiPgogICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-code-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBzaGFwZS1yZW5kZXJpbmc9Imdlb21ldHJpY1ByZWNpc2lvbiI+CiAgICA8cGF0aCBkPSJNNi41OSwzLjQxTDIsOEw2LjU5LDEyLjZMOCwxMS4xOEw0LjgyLDhMOCw0LjgyTDYuNTksMy40MU0xMi40MSwzLjQxTDExLDQuODJMMTQuMTgsOEwxMSwxMS4xOEwxMi40MSwxMi42TDE3LDhMMTIuNDEsMy40MU0yMS41OSwxMS41OUwxMy41LDE5LjY4TDkuODMsMTZMOC40MiwxNy40MUwxMy41LDIyLjVMMjMsMTNMMjEuNTksMTEuNTlaIiAvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-code: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTExLjQgMTguNkw2LjggMTRMMTEuNCA5LjRMMTAgOEw0IDE0TDEwIDIwTDExLjQgMTguNlpNMTYuNiAxOC42TDIxLjIgMTRMMTYuNiA5LjRMMTggOEwyNCAxNEwxOCAyMEwxNi42IDE4LjZWMTguNloiLz4KCTwvZz4KPC9zdmc+Cg==);
  --jp-icon-collapse-all: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTggMmMxIDAgMTEgMCAxMiAwczIgMSAyIDJjMCAxIDAgMTEgMCAxMnMwIDItMiAyQzIwIDE0IDIwIDQgMjAgNFMxMCA0IDYgNGMwLTIgMS0yIDItMnoiIC8+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTE4IDhjMC0xLTEtMi0yLTJTNSA2IDQgNnMtMiAxLTIgMmMwIDEgMCAxMSAwIDEyczEgMiAyIDJjMSAwIDExIDAgMTIgMHMyLTEgMi0yYzAtMSAwLTExIDAtMTJ6bS0yIDB2MTJINFY4eiIgLz4KICAgICAgICA8cGF0aCBkPSJNNiAxM3YyaDh2LTJ6IiAvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-console: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwMCAyMDAiPgogIDxnIGNsYXNzPSJqcC1jb25zb2xlLWljb24tYmFja2dyb3VuZC1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMjg4RDEiPgogICAgPHBhdGggZD0iTTIwIDE5LjhoMTYwdjE1OS45SDIweiIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtY29uc29sZS1pY29uLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIj4KICAgIDxwYXRoIGQ9Ik0xMDUgMTI3LjNoNDB2MTIuOGgtNDB6TTUxLjEgNzdMNzQgOTkuOWwtMjMuMyAyMy4zIDEwLjUgMTAuNSAyMy4zLTIzLjNMOTUgOTkuOSA4NC41IDg5LjQgNjEuNiA2Ni41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-copy: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTExLjksMUgzLjJDMi40LDEsMS43LDEuNywxLjcsMi41djEwLjJoMS41VjIuNWg4LjdWMXogTTE0LjEsMy45aC04Yy0wLjgsMC0xLjUsMC43LTEuNSwxLjV2MTAuMmMwLDAuOCwwLjcsMS41LDEuNSwxLjVoOCBjMC44LDAsMS41LTAuNywxLjUtMS41VjUuNEMxNS41LDQuNiwxNC45LDMuOSwxNC4xLDMuOXogTTE0LjEsMTUuNWgtOFY1LjRoOFYxNS41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-copyright: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAwIDI0IDI0IiBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCI+CiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0xMS44OCw5LjE0YzEuMjgsMC4wNiwxLjYxLDEuMTUsMS42MywxLjY2aDEuNzljLTAuMDgtMS45OC0xLjQ5LTMuMTktMy40NS0zLjE5QzkuNjQsNy42MSw4LDksOCwxMi4xNCBjMCwxLjk0LDAuOTMsNC4yNCwzLjg0LDQuMjRjMi4yMiwwLDMuNDEtMS42NSwzLjQ0LTIuOTVoLTEuNzljLTAuMDMsMC41OS0wLjQ1LDEuMzgtMS42MywxLjQ0QzEwLjU1LDE0LjgzLDEwLDEzLjgxLDEwLDEyLjE0IEMxMCw5LjI1LDExLjI4LDkuMTYsMTEuODgsOS4xNHogTTEyLDJDNi40OCwyLDIsNi40OCwyLDEyczQuNDgsMTAsMTAsMTBzMTAtNC40OCwxMC0xMFMxNy41MiwyLDEyLDJ6IE0xMiwyMGMtNC40MSwwLTgtMy41OS04LTggczMuNTktOCw4LThzOCwzLjU5LDgsOFMxNi40MSwyMCwxMiwyMHoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-cut: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkuNjQgNy42NGMuMjMtLjUuMzYtMS4wNS4zNi0xLjY0IDAtMi4yMS0xLjc5LTQtNC00UzIgMy43OSAyIDZzMS43OSA0IDQgNGMuNTkgMCAxLjE0LS4xMyAxLjY0LS4zNkwxMCAxMmwtMi4zNiAyLjM2QzcuMTQgMTQuMTMgNi41OSAxNCA2IDE0Yy0yLjIxIDAtNCAxLjc5LTQgNHMxLjc5IDQgNCA0IDQtMS43OSA0LTRjMC0uNTktLjEzLTEuMTQtLjM2LTEuNjRMMTIgMTRsNyA3aDN2LTFMOS42NCA3LjY0ek02IDhjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTAgMTJjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTYtNy41Yy0uMjggMC0uNS0uMjItLjUtLjVzLjIyLS41LjUtLjUuNS4yMi41LjUtLjIyLjUtLjUuNXpNMTkgM2wtNiA2IDIgMiA3LTdWM3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-delete: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2cHgiIGhlaWdodD0iMTZweCI+CiAgICA8cGF0aCBkPSJNMCAwaDI0djI0SDB6IiBmaWxsPSJub25lIiAvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjI2MjYyIiBkPSJNNiAxOWMwIDEuMS45IDIgMiAyaDhjMS4xIDAgMi0uOSAyLTJWN0g2djEyek0xOSA0aC0zLjVsLTEtMWgtNWwtMSAxSDV2MmgxNFY0eiIgLz4KPC9zdmc+Cg==);
  --jp-icon-download: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDloLTRWM0g5djZINWw3IDcgNy03ek01IDE4djJoMTR2LTJINXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-duplicate: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTIuNzk5OTggMC44NzVIOC44OTU4MkM5LjIwMDYxIDAuODc1IDkuNDQ5OTggMS4xMzkxNCA5LjQ0OTk4IDEuNDYxOThDOS40NDk5OCAxLjc4NDgyIDkuMjAwNjEgMi4wNDg5NiA4Ljg5NTgyIDIuMDQ4OTZIMy4zNTQxNUMzLjA0OTM2IDIuMDQ4OTYgMi43OTk5OCAyLjMxMzEgMi43OTk5OCAyLjYzNTk0VjkuNjc5NjlDMi43OTk5OCAxMC4wMDI1IDIuNTUwNjEgMTAuMjY2NyAyLjI0NTgyIDEwLjI2NjdDMS45NDEwMyAxMC4yNjY3IDEuNjkxNjUgMTAuMDAyNSAxLjY5MTY1IDkuNjc5NjlWMi4wNDg5NkMxLjY5MTY1IDEuNDAzMjggMi4xOTA0IDAuODc1IDIuNzk5OTggMC44NzVaTTUuMzY2NjUgMTEuOVY0LjU1SDExLjA4MzNWMTEuOUg1LjM2NjY1Wk00LjE0MTY1IDQuMTQxNjdDNC4xNDE2NSAzLjY5MDYzIDQuNTA3MjggMy4zMjUgNC45NTgzMiAzLjMyNUgxMS40OTE3QzExLjk0MjcgMy4zMjUgMTIuMzA4MyAzLjY5MDYzIDEyLjMwODMgNC4xNDE2N1YxMi4zMDgzQzEyLjMwODMgMTIuNzU5NCAxMS45NDI3IDEzLjEyNSAxMS40OTE3IDEzLjEyNUg0Ljk1ODMyQzQuNTA3MjggMTMuMTI1IDQuMTQxNjUgMTIuNzU5NCA0LjE0MTY1IDEyLjMwODNWNC4xNDE2N1oiIGZpbGw9IiM2MTYxNjEiLz4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNOS40MzU3NCA4LjI2NTA3SDguMzY0MzFWOS4zMzY1QzguMzY0MzEgOS40NTQzNSA4LjI2Nzg4IDkuNTUwNzggOC4xNTAwMiA5LjU1MDc4QzguMDMyMTcgOS41NTA3OCA3LjkzNTc0IDkuNDU0MzUgNy45MzU3NCA5LjMzNjVWOC4yNjUwN0g2Ljg2NDMxQzYuNzQ2NDUgOC4yNjUwNyA2LjY1MDAyIDguMTY4NjQgNi42NTAwMiA4LjA1MDc4QzYuNjUwMDIgNy45MzI5MiA2Ljc0NjQ1IDcuODM2NSA2Ljg2NDMxIDcuODM2NUg3LjkzNTc0VjYuNzY1MDdDNy45MzU3NCA2LjY0NzIxIDguMDMyMTcgNi41NTA3OCA4LjE1MDAyIDYuNTUwNzhDOC4yNjc4OCA2LjU1MDc4IDguMzY0MzEgNi42NDcyMSA4LjM2NDMxIDYuNzY1MDdWNy44MzY1SDkuNDM1NzRDOS41NTM2IDcuODM2NSA5LjY1MDAyIDcuOTMyOTIgOS42NTAwMiA4LjA1MDc4QzkuNjUwMDIgOC4xNjg2NCA5LjU1MzYgOC4yNjUwNyA5LjQzNTc0IDguMjY1MDdaIiBmaWxsPSIjNjE2MTYxIiBzdHJva2U9IiM2MTYxNjEiIHN0cm9rZS13aWR0aD0iMC41Ii8+Cjwvc3ZnPgo=);
  --jp-icon-edit: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMgMTcuMjVWMjFoMy43NUwxNy44MSA5Ljk0bC0zLjc1LTMuNzVMMyAxNy4yNXpNMjAuNzEgNy4wNGMuMzktLjM5LjM5LTEuMDIgMC0xLjQxbC0yLjM0LTIuMzRjLS4zOS0uMzktMS4wMi0uMzktMS40MSAwbC0xLjgzIDEuODMgMy43NSAzLjc1IDEuODMtMS44M3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-ellipses: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iNSIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxOSIgY3k9IjEyIiByPSIyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-error: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjE5IiByPSIyIi8+PHBhdGggZD0iTTEwIDNoNHYxMmgtNHoiLz48L2c+CjxwYXRoIGZpbGw9Im5vbmUiIGQ9Ik0wIDBoMjR2MjRIMHoiLz4KPC9zdmc+Cg==);
  --jp-icon-expand-all: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTggMmMxIDAgMTEgMCAxMiAwczIgMSAyIDJjMCAxIDAgMTEgMCAxMnMwIDItMiAyQzIwIDE0IDIwIDQgMjAgNFMxMCA0IDYgNGMwLTIgMS0yIDItMnoiIC8+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTE4IDhjMC0xLTEtMi0yLTJTNSA2IDQgNnMtMiAxLTIgMmMwIDEgMCAxMSAwIDEyczEgMiAyIDJjMSAwIDExIDAgMTIgMHMyLTEgMi0yYzAtMSAwLTExIDAtMTJ6bS0yIDB2MTJINFY4eiIgLz4KICAgICAgICA8cGF0aCBkPSJNMTEgMTBIOXYzSDZ2MmgzdjNoMnYtM2gzdi0yaC0zeiIgLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-extension: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwLjUgMTFIMTlWN2MwLTEuMS0uOS0yLTItMmgtNFYzLjVDMTMgMi4xMiAxMS44OCAxIDEwLjUgMVM4IDIuMTIgOCAzLjVWNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAydjMuOEgzLjVjMS40OSAwIDIuNyAxLjIxIDIuNyAyLjdzLTEuMjEgMi43LTIuNyAyLjdIMlYyMGMwIDEuMS45IDIgMiAyaDMuOHYtMS41YzAtMS40OSAxLjIxLTIuNyAyLjctMi43IDEuNDkgMCAyLjcgMS4yMSAyLjcgMi43VjIySDE3YzEuMSAwIDItLjkgMi0ydi00aDEuNWMxLjM4IDAgMi41LTEuMTIgMi41LTIuNVMyMS44OCAxMSAyMC41IDExeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-fast-forward: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTQgMThsOC41LTZMNCA2djEyem05LTEydjEybDguNS02TDEzIDZ6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-file-upload: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkgMTZoNnYtNmg0bC03LTctNyA3aDR6bS00IDJoMTR2Mkg1eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-file: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuMyA4LjJsLTUuNS01LjVjLS4zLS4zLS43LS41LTEuMi0uNUgzLjljLS44LjEtMS42LjktMS42IDEuOHYxNC4xYzAgLjkuNyAxLjYgMS42IDEuNmgxNC4yYy45IDAgMS42LS43IDEuNi0xLjZWOS40Yy4xLS41LS4xLS45LS40LTEuMnptLTUuOC0zLjNsMy40IDMuNmgtMy40VjQuOXptMy45IDEyLjdINC43Yy0uMSAwLS4yIDAtLjItLjJWNC43YzAtLjIuMS0uMy4yLS4zaDcuMnY0LjRzMCAuOC4zIDEuMWMuMy4zIDEuMS4zIDEuMS4zaDQuM3Y3LjJzLS4xLjItLjIuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-filter-dot: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTE0LDEyVjE5Ljg4QzE0LjA0LDIwLjE4IDEzLjk0LDIwLjUgMTMuNzEsMjAuNzFDMTMuMzIsMjEuMSAxMi42OSwyMS4xIDEyLjMsMjAuNzFMMTAuMjksMTguN0MxMC4wNiwxOC40NyA5Ljk2LDE4LjE2IDEwLDE3Ljg3VjEySDkuOTdMNC4yMSw0LjYyQzMuODcsNC4xOSAzLjk1LDMuNTYgNC4zOCwzLjIyQzQuNTcsMy4wOCA0Ljc4LDMgNSwzVjNIMTlWM0MxOS4yMiwzIDE5LjQzLDMuMDggMTkuNjIsMy4yMkMyMC4wNSwzLjU2IDIwLjEzLDQuMTkgMTkuNzksNC42MkwxNC4wMywxMkgxNFoiIC8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWRvdCIgZmlsbD0iI0ZGRiI+CiAgICA8Y2lyY2xlIGN4PSIxOCIgY3k9IjE3IiByPSIzIj48L2NpcmNsZT4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-filter-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEwIDE4aDR2LTJoLTR2MnpNMyA2djJoMThWNkgzem0zIDdoMTJ2LTJINnYyeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-filter: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTE0LDEyVjE5Ljg4QzE0LjA0LDIwLjE4IDEzLjk0LDIwLjUgMTMuNzEsMjAuNzFDMTMuMzIsMjEuMSAxMi42OSwyMS4xIDEyLjMsMjAuNzFMMTAuMjksMTguN0MxMC4wNiwxOC40NyA5Ljk2LDE4LjE2IDEwLDE3Ljg3VjEySDkuOTdMNC4yMSw0LjYyQzMuODcsNC4xOSAzLjk1LDMuNTYgNC4zOCwzLjIyQzQuNTcsMy4wOCA0Ljc4LDMgNSwzVjNIMTlWM0MxOS4yMiwzIDE5LjQzLDMuMDggMTkuNjIsMy4yMkMyMC4wNSwzLjU2IDIwLjEzLDQuMTkgMTkuNzksNC42MkwxNC4wMywxMkgxNFoiIC8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-folder-favorite: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMjRweCIgZmlsbD0iIzAwMDAwMCI+CiAgPHBhdGggZD0iTTAgMGgyNHYyNEgwVjB6IiBmaWxsPSJub25lIi8+PHBhdGggY2xhc3M9ImpwLWljb24zIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxNjE2MSIgZD0iTTIwIDZoLThsLTItMkg0Yy0xLjEgMC0yIC45LTIgMnYxMmMwIDEuMS45IDIgMiAyaDE2YzEuMSAwIDItLjkgMi0yVjhjMC0xLjEtLjktMi0yLTJ6bS0yLjA2IDExTDE1IDE1LjI4IDEyLjA2IDE3bC43OC0zLjMzLTIuNTktMi4yNCAzLjQxLS4yOUwxNSA4bDEuMzQgMy4xNCAzLjQxLjI5LTIuNTkgMi4yNC43OCAzLjMzeiIvPgo8L3N2Zz4K);
  --jp-icon-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY4YzAtMS4xLS45LTItMi0yaC04bC0yLTJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-home: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMjRweCIgZmlsbD0iIzAwMDAwMCI+CiAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGNsYXNzPSJqcC1pY29uMyBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xMCAyMHYtNmg0djZoNXYtOGgzTDEyIDMgMiAxMmgzdjh6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-html5: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMDAiIGQ9Ik0xMDguNCAwaDIzdjIyLjhoMjEuMlYwaDIzdjY5aC0yM1Y0NmgtMjF2MjNoLTIzLjJNMjA2IDIzaC0yMC4zVjBoNjMuN3YyM0gyMjl2NDZoLTIzbTUzLjUtNjloMjQuMWwxNC44IDI0LjNMMzEzLjIgMGgyNC4xdjY5aC0yM1YzNC44bC0xNi4xIDI0LjgtMTYuMS0yNC44VjY5aC0yMi42bTg5LjItNjloMjN2NDYuMmgzMi42VjY5aC01NS42Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2U0NGQyNiIgZD0iTTEwNy42IDQ3MWwtMzMtMzcwLjRoMzYyLjhsLTMzIDM3MC4yTDI1NS43IDUxMiIvPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNmMTY1MjkiIGQ9Ik0yNTYgNDgwLjVWMTMxaDE0OC4zTDM3NiA0NDciLz4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNlYmViZWIiIGQ9Ik0xNDIgMTc2LjNoMTE0djQ1LjRoLTY0LjJsNC4yIDQ2LjVoNjB2NDUuM0gxNTQuNG0yIDIyLjhIMjAybDMuMiAzNi4zIDUwLjggMTMuNnY0Ny40bC05My4yLTI2Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIiBkPSJNMzY5LjYgMTc2LjNIMjU1Ljh2NDUuNGgxMDkuNm0tNC4xIDQ2LjVIMjU1Ljh2NDUuNGg1NmwtNS4zIDU5LTUwLjcgMTMuNnY0Ny4ybDkzLTI1LjgiLz4KPC9zdmc+Cg==);
  --jp-icon-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1icmFuZDQganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNGRkYiIGQ9Ik0yLjIgMi4yaDE3LjV2MTcuNUgyLjJ6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzNGNTFCNSIgZD0iTTIuMiAyLjJ2MTcuNWgxNy41bC4xLTE3LjVIMi4yem0xMi4xIDIuMmMxLjIgMCAyLjIgMSAyLjIgMi4ycy0xIDIuMi0yLjIgMi4yLTIuMi0xLTIuMi0yLjIgMS0yLjIgMi4yLTIuMnpNNC40IDE3LjZsMy4zLTguOCAzLjMgNi42IDIuMi0zLjIgNC40IDUuNEg0LjR6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-info: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUwLjk3OCA1MC45NzgiPgoJPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KCQk8cGF0aCBkPSJNNDMuNTIsNy40NThDMzguNzExLDIuNjQ4LDMyLjMwNywwLDI1LjQ4OSwwQzE4LjY3LDAsMTIuMjY2LDIuNjQ4LDcuNDU4LDcuNDU4CgkJCWMtOS45NDMsOS45NDEtOS45NDMsMjYuMTE5LDAsMzYuMDYyYzQuODA5LDQuODA5LDExLjIxMiw3LjQ1NiwxOC4wMzEsNy40NThjMCwwLDAuMDAxLDAsMC4wMDIsMAoJCQljNi44MTYsMCwxMy4yMjEtMi42NDgsMTguMDI5LTcuNDU4YzQuODA5LTQuODA5LDcuNDU3LTExLjIxMiw3LjQ1Ny0xOC4wM0M1MC45NzcsMTguNjcsNDguMzI4LDEyLjI2Niw0My41Miw3LjQ1OHoKCQkJIE00Mi4xMDYsNDIuMTA1Yy00LjQzMiw0LjQzMS0xMC4zMzIsNi44NzItMTYuNjE1LDYuODcyaC0wLjAwMmMtNi4yODUtMC4wMDEtMTIuMTg3LTIuNDQxLTE2LjYxNy02Ljg3MgoJCQljLTkuMTYyLTkuMTYzLTkuMTYyLTI0LjA3MSwwLTMzLjIzM0MxMy4zMDMsNC40NCwxOS4yMDQsMiwyNS40ODksMmM2LjI4NCwwLDEyLjE4NiwyLjQ0LDE2LjYxNyw2Ljg3MgoJCQljNC40MzEsNC40MzEsNi44NzEsMTAuMzMyLDYuODcxLDE2LjYxN0M0OC45NzcsMzEuNzcyLDQ2LjUzNiwzNy42NzUsNDIuMTA2LDQyLjEwNXoiLz4KCQk8cGF0aCBkPSJNMjMuNTc4LDMyLjIxOGMtMC4wMjMtMS43MzQsMC4xNDMtMy4wNTksMC40OTYtMy45NzJjMC4zNTMtMC45MTMsMS4xMS0xLjk5NywyLjI3Mi0zLjI1MwoJCQljMC40NjgtMC41MzYsMC45MjMtMS4wNjIsMS4zNjctMS41NzVjMC42MjYtMC43NTMsMS4xMDQtMS40NzgsMS40MzYtMi4xNzVjMC4zMzEtMC43MDcsMC40OTUtMS41NDEsMC40OTUtMi41CgkJCWMwLTEuMDk2LTAuMjYtMi4wODgtMC43NzktMi45NzljLTAuNTY1LTAuODc5LTEuNTAxLTEuMzM2LTIuODA2LTEuMzY5Yy0xLjgwMiwwLjA1Ny0yLjk4NSwwLjY2Ny0zLjU1LDEuODMyCgkJCWMtMC4zMDEsMC41MzUtMC41MDMsMS4xNDEtMC42MDcsMS44MTRjLTAuMTM5LDAuNzA3LTAuMjA3LDEuNDMyLTAuMjA3LDIuMTc0aC0yLjkzN2MtMC4wOTEtMi4yMDgsMC40MDctNC4xMTQsMS40OTMtNS43MTkKCQkJYzEuMDYyLTEuNjQsMi44NTUtMi40ODEsNS4zNzgtMi41MjdjMi4xNiwwLjAyMywzLjg3NCwwLjYwOCw1LjE0MSwxLjc1OGMxLjI3OCwxLjE2LDEuOTI5LDIuNzY0LDEuOTUsNC44MTEKCQkJYzAsMS4xNDItMC4xMzcsMi4xMTEtMC40MSwyLjkxMWMtMC4zMDksMC44NDUtMC43MzEsMS41OTMtMS4yNjgsMi4yNDNjLTAuNDkyLDAuNjUtMS4wNjgsMS4zMTgtMS43MywyLjAwMgoJCQljLTAuNjUsMC42OTctMS4zMTMsMS40NzktMS45ODcsMi4zNDZjLTAuMjM5LDAuMzc3LTAuNDI5LDAuNzc3LTAuNTY1LDEuMTk5Yy0wLjE2LDAuOTU5LTAuMjE3LDEuOTUxLTAuMTcxLDIuOTc5CgkJCUMyNi41ODksMzIuMjE4LDIzLjU3OCwzMi4yMTgsMjMuNTc4LDMyLjIxOHogTTIzLjU3OCwzOC4yMnYtMy40ODRoMy4wNzZ2My40ODRIMjMuNTc4eiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-inspector: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaW5zcGVjdG9yLWljb24tY29sb3IganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY2YzAtMS4xLS45LTItMi0yem0tNSAxNEg0di00aDExdjR6bTAtNUg0VjloMTF2NHptNSA1aC00VjloNHY5eiIvPgo8L3N2Zz4K);
  --jp-icon-json: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtanNvbi1pY29uLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI0Y5QTgyNSI+CiAgICA8cGF0aCBkPSJNMjAuMiAxMS44Yy0xLjYgMC0xLjcuNS0xLjcgMSAwIC40LjEuOS4xIDEuMy4xLjUuMS45LjEgMS4zIDAgMS43LTEuNCAyLjMtMy41IDIuM2gtLjl2LTEuOWguNWMxLjEgMCAxLjQgMCAxLjQtLjggMC0uMyAwLS42LS4xLTEgMC0uNC0uMS0uOC0uMS0xLjIgMC0xLjMgMC0xLjggMS4zLTItMS4zLS4yLTEuMy0uNy0xLjMtMiAwLS40LjEtLjguMS0xLjIuMS0uNC4xLS43LjEtMSAwLS44LS40LS43LTEuNC0uOGgtLjVWNC4xaC45YzIuMiAwIDMuNS43IDMuNSAyLjMgMCAuNC0uMS45LS4xIDEuMy0uMS41LS4xLjktLjEgMS4zIDAgLjUuMiAxIDEuNyAxdjEuOHpNMS44IDEwLjFjMS42IDAgMS43LS41IDEuNy0xIDAtLjQtLjEtLjktLjEtMS4zLS4xLS41LS4xLS45LS4xLTEuMyAwLTEuNiAxLjQtMi4zIDMuNS0yLjNoLjl2MS45aC0uNWMtMSAwLTEuNCAwLTEuNC44IDAgLjMgMCAuNi4xIDEgMCAuMi4xLjYuMSAxIDAgMS4zIDAgMS44LTEuMyAyQzYgMTEuMiA2IDExLjcgNiAxM2MwIC40LS4xLjgtLjEgMS4yLS4xLjMtLjEuNy0uMSAxIDAgLjguMy44IDEuNC44aC41djEuOWgtLjljLTIuMSAwLTMuNS0uNi0zLjUtMi4zIDAtLjQuMS0uOS4xLTEuMy4xLS41LjEtLjkuMS0xLjMgMC0uNS0uMi0xLTEuNy0xdi0xLjl6Ii8+CiAgICA8Y2lyY2xlIGN4PSIxMSIgY3k9IjEzLjgiIHI9IjIuMSIvPgogICAgPGNpcmNsZSBjeD0iMTEiIGN5PSI4LjIiIHI9IjIuMSIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-julia: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDMyNSAzMDAiPgogIDxnIGNsYXNzPSJqcC1icmFuZDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjY2IzYzMzIj4KICAgIDxwYXRoIGQ9Ik0gMTUwLjg5ODQzOCAyMjUgQyAxNTAuODk4NDM4IDI2Ni40MjE4NzUgMTE3LjMyMDMxMiAzMDAgNzUuODk4NDM4IDMwMCBDIDM0LjQ3NjU2MiAzMDAgMC44OTg0MzggMjY2LjQyMTg3NSAwLjg5ODQzOCAyMjUgQyAwLjg5ODQzOCAxODMuNTc4MTI1IDM0LjQ3NjU2MiAxNTAgNzUuODk4NDM4IDE1MCBDIDExNy4zMjAzMTIgMTUwIDE1MC44OTg0MzggMTgzLjU3ODEyNSAxNTAuODk4NDM4IDIyNSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzM4OTgyNiI+CiAgICA8cGF0aCBkPSJNIDIzNy41IDc1IEMgMjM3LjUgMTE2LjQyMTg3NSAyMDMuOTIxODc1IDE1MCAxNjIuNSAxNTAgQyAxMjEuMDc4MTI1IDE1MCA4Ny41IDExNi40MjE4NzUgODcuNSA3NSBDIDg3LjUgMzMuNTc4MTI1IDEyMS4wNzgxMjUgMCAxNjIuNSAwIEMgMjAzLjkyMTg3NSAwIDIzNy41IDMzLjU3ODEyNSAyMzcuNSA3NSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzk1NThiMiI+CiAgICA8cGF0aCBkPSJNIDMyNC4xMDE1NjIgMjI1IEMgMzI0LjEwMTU2MiAyNjYuNDIxODc1IDI5MC41MjM0MzggMzAwIDI0OS4xMDE1NjIgMzAwIEMgMjA3LjY3OTY4OCAzMDAgMTc0LjEwMTU2MiAyNjYuNDIxODc1IDE3NC4xMDE1NjIgMjI1IEMgMTc0LjEwMTU2MiAxODMuNTc4MTI1IDIwNy42Nzk2ODggMTUwIDI0OS4xMDE1NjIgMTUwIEMgMjkwLjUyMzQzOCAxNTAgMzI0LjEwMTU2MiAxODMuNTc4MTI1IDMyNC4xMDE1NjIgMjI1Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-jupyter-favicon: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUyIiBoZWlnaHQ9IjE2NSIgdmlld0JveD0iMCAwIDE1MiAxNjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgPGcgY2xhc3M9ImpwLWp1cHl0ZXItaWNvbi1jb2xvciIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA3ODk0NywgMTEwLjU4MjkyNykiIGQ9Ik03NS45NDIyODQyLDI5LjU4MDQ1NjEgQzQzLjMwMjM5NDcsMjkuNTgwNDU2MSAxNC43OTY3ODMyLDE3LjY1MzQ2MzQgMCwwIEM1LjUxMDgzMjExLDE1Ljg0MDY4MjkgMTUuNzgxNTM4OSwyOS41NjY3NzMyIDI5LjM5MDQ5NDcsMzkuMjc4NDE3MSBDNDIuOTk5Nyw0OC45ODk4NTM3IDU5LjI3MzcsNTQuMjA2NzgwNSA3NS45NjA1Nzg5LDU0LjIwNjc4MDUgQzkyLjY0NzQ1NzksNTQuMjA2NzgwNSAxMDguOTIxNDU4LDQ4Ljk4OTg1MzcgMTIyLjUzMDY2MywzOS4yNzg0MTcxIEMxMzYuMTM5NDUzLDI5LjU2Njc3MzIgMTQ2LjQxMDI4NCwxNS44NDA2ODI5IDE1MS45MjExNTgsMCBDMTM3LjA4Nzg2OCwxNy42NTM0NjM0IDEwOC41ODI1ODksMjkuNTgwNDU2MSA3NS45NDIyODQyLDI5LjU4MDQ1NjEgTDc1Ljk0MjI4NDIsMjkuNTgwNDU2MSBaIiAvPgogICAgPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMzczNjgsIDAuNzA0ODc4KSIgZD0iTTc1Ljk3ODQ1NzksMjQuNjI2NDA3MyBDMTA4LjYxODc2MywyNC42MjY0MDczIDEzNy4xMjQ0NTgsMzYuNTUzNDQxNSAxNTEuOTIxMTU4LDU0LjIwNjc4MDUgQzE0Ni40MTAyODQsMzguMzY2MjIyIDEzNi4xMzk0NTMsMjQuNjQwMTMxNyAxMjIuNTMwNjYzLDE0LjkyODQ4NzggQzEwOC45MjE0NTgsNS4yMTY4NDM5IDkyLjY0NzQ1NzksMCA3NS45NjA1Nzg5LDAgQzU5LjI3MzcsMCA0Mi45OTk3LDUuMjE2ODQzOSAyOS4zOTA0OTQ3LDE0LjkyODQ4NzggQzE1Ljc4MTUzODksMjQuNjQwMTMxNyA1LjUxMDgzMjExLDM4LjM2NjIyMiAwLDU0LjIwNjc4MDUgQzE0LjgzMzA4MTYsMzYuNTg5OTI5MyA0My4zMzg1Njg0LDI0LjYyNjQwNzMgNzUuOTc4NDU3OSwyNC42MjY0MDczIEw3NS45Nzg0NTc5LDI0LjYyNjQwNzMgWiIgLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-jupyter: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzkiIGhlaWdodD0iNTEiIHZpZXdCb3g9IjAgMCAzOSA1MSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMTYzOCAtMjI4MSkiPgogICAgIDxnIGNsYXNzPSJqcC1qdXB5dGVyLWljb24tY29sb3IiIGZpbGw9IiNGMzc3MjYiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5Ljc0IDIzMTEuOTgpIiBkPSJNIDE4LjI2NDYgNy4xMzQxMUMgMTAuNDE0NSA3LjEzNDExIDMuNTU4NzIgNC4yNTc2IDAgMEMgMS4zMjUzOSAzLjgyMDQgMy43OTU1NiA3LjEzMDgxIDcuMDY4NiA5LjQ3MzAzQyAxMC4zNDE3IDExLjgxNTIgMTQuMjU1NyAxMy4wNzM0IDE4LjI2OSAxMy4wNzM0QyAyMi4yODIzIDEzLjA3MzQgMjYuMTk2MyAxMS44MTUyIDI5LjQ2OTQgOS40NzMwM0MgMzIuNzQyNCA3LjEzMDgxIDM1LjIxMjYgMy44MjA0IDM2LjUzOCAwQyAzMi45NzA1IDQuMjU3NiAyNi4xMTQ4IDcuMTM0MTEgMTguMjY0NiA3LjEzNDExWiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5LjczIDIyODUuNDgpIiBkPSJNIDE4LjI3MzMgNS45MzkzMUMgMjYuMTIzNSA1LjkzOTMxIDMyLjk3OTMgOC44MTU4MyAzNi41MzggMTMuMDczNEMgMzUuMjEyNiA5LjI1MzAzIDMyLjc0MjQgNS45NDI2MiAyOS40Njk0IDMuNjAwNEMgMjYuMTk2MyAxLjI1ODE4IDIyLjI4MjMgMCAxOC4yNjkgMEMgMTQuMjU1NyAwIDEwLjM0MTcgMS4yNTgxOCA3LjA2ODYgMy42MDA0QyAzLjc5NTU2IDUuOTQyNjIgMS4zMjUzOSA5LjI1MzAzIDAgMTMuMDczNEMgMy41Njc0NSA4LjgyNDYzIDEwLjQyMzIgNS45MzkzMSAxOC4yNzMzIDUuOTM5MzFaIi8+CiAgICA8L2c+CiAgICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjY5LjMgMjI4MS4zMSkiIGQ9Ik0gNS44OTM1MyAyLjg0NEMgNS45MTg4OSAzLjQzMTY1IDUuNzcwODUgNC4wMTM2NyA1LjQ2ODE1IDQuNTE2NDVDIDUuMTY1NDUgNS4wMTkyMiA0LjcyMTY4IDUuNDIwMTUgNC4xOTI5OSA1LjY2ODUxQyAzLjY2NDMgNS45MTY4OCAzLjA3NDQ0IDYuMDAxNTEgMi40OTgwNSA1LjkxMTcxQyAxLjkyMTY2IDUuODIxOSAxLjM4NDYzIDUuNTYxNyAwLjk1NDg5OCA1LjE2NDAxQyAwLjUyNTE3IDQuNzY2MzMgMC4yMjIwNTYgNC4yNDkwMyAwLjA4MzkwMzcgMy42Nzc1N0MgLTAuMDU0MjQ4MyAzLjEwNjExIC0wLjAyMTIzIDIuNTA2MTcgMC4xNzg3ODEgMS45NTM2NEMgMC4zNzg3OTMgMS40MDExIDAuNzM2ODA5IDAuOTIwODE3IDEuMjA3NTQgMC41NzM1MzhDIDEuNjc4MjYgMC4yMjYyNTkgMi4yNDA1NSAwLjAyNzU5MTkgMi44MjMyNiAwLjAwMjY3MjI5QyAzLjYwMzg5IC0wLjAzMDcxMTUgNC4zNjU3MyAwLjI0OTc4OSA0Ljk0MTQyIDAuNzgyNTUxQyA1LjUxNzExIDEuMzE1MzEgNS44NTk1NiAyLjA1Njc2IDUuODkzNTMgMi44NDRaIi8+CiAgICAgIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE2MzkuOCAyMzIzLjgxKSIgZD0iTSA3LjQyNzg5IDMuNTgzMzhDIDcuNDYwMDggNC4zMjQzIDcuMjczNTUgNS4wNTgxOSA2Ljg5MTkzIDUuNjkyMTNDIDYuNTEwMzEgNi4zMjYwNyA1Ljk1MDc1IDYuODMxNTYgNS4yODQxMSA3LjE0NDZDIDQuNjE3NDcgNy40NTc2MyAzLjg3MzcxIDcuNTY0MTQgMy4xNDcwMiA3LjQ1MDYzQyAyLjQyMDMyIDcuMzM3MTIgMS43NDMzNiA3LjAwODcgMS4yMDE4NCA2LjUwNjk1QyAwLjY2MDMyOCA2LjAwNTIgMC4yNzg2MSA1LjM1MjY4IDAuMTA1MDE3IDQuNjMyMDJDIC0wLjA2ODU3NTcgMy45MTEzNSAtMC4wMjYyMzYxIDMuMTU0OTQgMC4yMjY2NzUgMi40NTg1NkMgMC40Nzk1ODcgMS43NjIxNyAwLjkzMTY5NyAxLjE1NzEzIDEuNTI1NzYgMC43MjAwMzNDIDIuMTE5ODMgMC4yODI5MzUgMi44MjkxNCAwLjAzMzQzOTUgMy41NjM4OSAwLjAwMzEzMzQ0QyA0LjU0NjY3IC0wLjAzNzQwMzMgNS41MDUyOSAwLjMxNjcwNiA2LjIyOTYxIDAuOTg3ODM1QyA2Ljk1MzkzIDEuNjU4OTYgNy4zODQ4NCAyLjU5MjM1IDcuNDI3ODkgMy41ODMzOEwgNy40Mjc4OSAzLjU4MzM4WiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM4LjM2IDIyODYuMDYpIiBkPSJNIDIuMjc0NzEgNC4zOTYyOUMgMS44NDM2MyA0LjQxNTA4IDEuNDE2NzEgNC4zMDQ0NSAxLjA0Nzk5IDQuMDc4NDNDIDAuNjc5MjY4IDMuODUyNCAwLjM4NTMyOCAzLjUyMTE0IDAuMjAzMzcxIDMuMTI2NTZDIDAuMDIxNDEzNiAyLjczMTk4IC0wLjA0MDM3OTggMi4yOTE4MyAwLjAyNTgxMTYgMS44NjE4MUMgMC4wOTIwMDMxIDEuNDMxOCAwLjI4MzIwNCAxLjAzMTI2IDAuNTc1MjEzIDAuNzEwODgzQyAwLjg2NzIyMiAwLjM5MDUxIDEuMjQ2OTEgMC4xNjQ3MDggMS42NjYyMiAwLjA2MjA1OTJDIDIuMDg1NTMgLTAuMDQwNTg5NyAyLjUyNTYxIC0wLjAxNTQ3MTQgMi45MzA3NiAwLjEzNDIzNUMgMy4zMzU5MSAwLjI4Mzk0MSAzLjY4NzkyIDAuNTUxNTA1IDMuOTQyMjIgMC45MDMwNkMgNC4xOTY1MiAxLjI1NDYyIDQuMzQxNjkgMS42NzQzNiA0LjM1OTM1IDIuMTA5MTZDIDQuMzgyOTkgMi42OTEwNyA0LjE3Njc4IDMuMjU4NjkgMy43ODU5NyAzLjY4NzQ2QyAzLjM5NTE2IDQuMTE2MjQgMi44NTE2NiA0LjM3MTE2IDIuMjc0NzEgNC4zOTYyOUwgMi4yNzQ3MSA0LjM5NjI5WiIvPgogICAgPC9nPgogIDwvZz4+Cjwvc3ZnPgo=);
  --jp-icon-jupyterlab-wordmark: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIHZpZXdCb3g9IjAgMCAxODYwLjggNDc1Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0RTRFNEUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ4MC4xMzY0MDEsIDY0LjI3MTQ5MykiPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDU4Ljg3NTU2NikiPgogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA4NzYwMywgMC4xNDAyOTQpIj4KICAgICAgICA8cGF0aCBkPSJNLTQyNi45LDE2OS44YzAsNDguNy0zLjcsNjQuNy0xMy42LDc2LjRjLTEwLjgsMTAtMjUsMTUuNS0zOS43LDE1LjVsMy43LDI5IGMyMi44LDAuMyw0NC44LTcuOSw2MS45LTIzLjFjMTcuOC0xOC41LDI0LTQ0LjEsMjQtODMuM1YwSC00Mjd2MTcwLjFMLTQyNi45LDE2OS44TC00MjYuOSwxNjkuOHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTU1LjA0NTI5NiwgNTYuODM3MTA0KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTYyNDUzLCAxLjc5OTg0MikiPgogICAgICAgIDxwYXRoIGQ9Ik0tMzEyLDE0OGMwLDIxLDAsMzkuNSwxLjcsNTUuNGgtMzEuOGwtMi4xLTMzLjNoLTAuOGMtNi43LDExLjYtMTYuNCwyMS4zLTI4LDI3LjkgYy0xMS42LDYuNi0yNC44LDEwLTM4LjIsOS44Yy0zMS40LDAtNjktMTcuNy02OS04OVYwaDM2LjR2MTEyLjdjMCwzOC43LDExLjYsNjQuNyw0NC42LDY0LjdjMTAuMy0wLjIsMjAuNC0zLjUsMjguOS05LjQgYzguNS01LjksMTUuMS0xNC4zLDE4LjktMjMuOWMyLjItNi4xLDMuMy0xMi41LDMuMy0xOC45VjAuMmgzNi40VjE0OEgtMzEyTC0zMTIsMTQ4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzOTAuMDEzMzIyLCA1My40Nzk2MzgpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS43MDY0NTgsIDAuMjMxNDI1KSI+CiAgICAgICAgPHBhdGggZD0iTS00NzguNiw3MS40YzAtMjYtMC44LTQ3LTEuNy02Ni43aDMyLjdsMS43LDM0LjhoMC44YzcuMS0xMi41LDE3LjUtMjIuOCwzMC4xLTI5LjcgYzEyLjUtNywyNi43LTEwLjMsNDEtOS44YzQ4LjMsMCw4NC43LDQxLjcsODQuNywxMDMuM2MwLDczLjEtNDMuNywxMDkuMi05MSwxMDkuMmMtMTIuMSwwLjUtMjQuMi0yLjItMzUtNy44IGMtMTAuOC01LjYtMTkuOS0xMy45LTI2LjYtMjQuMmgtMC44VjI5MWgtMzZ2LTIyMEwtNDc4LjYsNzEuNEwtNDc4LjYsNzEuNHogTS00NDIuNiwxMjUuNmMwLjEsNS4xLDAuNiwxMC4xLDEuNywxNS4xIGMzLDEyLjMsOS45LDIzLjMsMTkuOCwzMS4xYzkuOSw3LjgsMjIuMSwxMi4xLDM0LjcsMTIuMWMzOC41LDAsNjAuNy0zMS45LDYwLjctNzguNWMwLTQwLjctMjEuMS03NS42LTU5LjUtNzUuNiBjLTEyLjksMC40LTI1LjMsNS4xLTM1LjMsMTMuNGMtOS45LDguMy0xNi45LDE5LjctMTkuNiwzMi40Yy0xLjUsNC45LTIuMywxMC0yLjUsMTUuMVYxMjUuNkwtNDQyLjYsMTI1LjZMLTQ0Mi42LDEyNS42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2MDYuNzQwNzI2LCA1Ni44MzcxMDQpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC43NTEyMjYsIDEuOTg5Mjk5KSI+CiAgICAgICAgPHBhdGggZD0iTS00NDAuOCwwbDQzLjcsMTIwLjFjNC41LDEzLjQsOS41LDI5LjQsMTIuOCw0MS43aDAuOGMzLjctMTIuMiw3LjktMjcuNywxMi44LTQyLjQgbDM5LjctMTE5LjJoMzguNUwtMzQ2LjksMTQ1Yy0yNiw2OS43LTQzLjcsMTA1LjQtNjguNiwxMjcuMmMtMTIuNSwxMS43LTI3LjksMjAtNDQuNiwyMy45bC05LjEtMzEuMSBjMTEuNy0zLjksMjIuNS0xMC4xLDMxLjgtMTguMWMxMy4yLTExLjEsMjMuNy0yNS4yLDMwLjYtNDEuMmMxLjUtMi44LDIuNS01LjcsMi45LTguOGMtMC4zLTMuMy0xLjItNi42LTIuNS05LjdMLTQ4MC4yLDAuMSBoMzkuN0wtNDQwLjgsMEwtNDQwLjgsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODIyLjc0ODEwNCwgMC4wMDAwMDApIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS40NjQwNTAsIDAuMzc4OTE0KSI+CiAgICAgICAgPHBhdGggZD0iTS00MTMuNywwdjU4LjNoNTJ2MjguMmgtNTJWMTk2YzAsMjUsNywzOS41LDI3LjMsMzkuNWM3LjEsMC4xLDE0LjItMC43LDIxLjEtMi41IGwxLjcsMjcuN2MtMTAuMywzLjctMjEuMyw1LjQtMzIuMiw1Yy03LjMsMC40LTE0LjYtMC43LTIxLjMtMy40Yy02LjgtMi43LTEyLjktNi44LTE3LjktMTIuMWMtMTAuMy0xMC45LTE0LjEtMjktMTQuMS01Mi45IFY4Ni41aC0zMVY1OC4zaDMxVjkuNkwtNDEzLjcsMEwtNDEzLjcsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOTc0LjQzMzI4NiwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuOTkwMDM0LCAwLjYxMDMzOSkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDQ1LjgsMTEzYzAuOCw1MCwzMi4yLDcwLjYsNjguNiw3MC42YzE5LDAuNiwzNy45LTMsNTUuMy0xMC41bDYuMiwyNi40IGMtMjAuOSw4LjktNDMuNSwxMy4xLTY2LjIsMTIuNmMtNjEuNSwwLTk4LjMtNDEuMi05OC4zLTEwMi41Qy00ODAuMiw0OC4yLTQ0NC43LDAtMzg2LjUsMGM2NS4yLDAsODIuNyw1OC4zLDgyLjcsOTUuNyBjLTAuMSw1LjgtMC41LDExLjUtMS4yLDE3LjJoLTE0MC42SC00NDUuOEwtNDQ1LjgsMTEzeiBNLTMzOS4yLDg2LjZjMC40LTIzLjUtOS41LTYwLjEtNTAuNC02MC4xIGMtMzYuOCwwLTUyLjgsMzQuNC01NS43LDYwLjFILTMzOS4yTC0zMzkuMiw4Ni42TC0zMzkuMiw4Ni42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjAxLjk2MTA1OCwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuMTc5NjQwLCAwLjcwNTA2OCkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDc4LjYsNjhjMC0yMy45LTAuNC00NC41LTEuNy02My40aDMxLjhsMS4yLDM5LjloMS43YzkuMS0yNy4zLDMxLTQ0LjUsNTUuMy00NC41IGMzLjUtMC4xLDcsMC40LDEwLjMsMS4ydjM0LjhjLTQuMS0wLjktOC4yLTEuMy0xMi40LTEuMmMtMjUuNiwwLTQzLjcsMTkuNy00OC43LDQ3LjRjLTEsNS43LTEuNiwxMS41LTEuNywxNy4ydjEwOC4zaC0zNlY2OCBMLTQ3OC42LDY4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCBkPSJNMTM1Mi4zLDMyNi4yaDM3VjI4aC0zN1YzMjYuMnogTTE2MDQuOCwzMjYuMmMtMi41LTEzLjktMy40LTMxLjEtMy40LTQ4Ljd2LTc2IGMwLTQwLjctMTUuMS04My4xLTc3LjMtODMuMWMtMjUuNiwwLTUwLDcuMS02Ni44LDE4LjFsOC40LDI0LjRjMTQuMy05LjIsMzQtMTUuMSw1My0xNS4xYzQxLjYsMCw0Ni4yLDMwLjIsNDYuMiw0N3Y0LjIgYy03OC42LTAuNC0xMjIuMywyNi41LTEyMi4zLDc1LjZjMCwyOS40LDIxLDU4LjQsNjIuMiw1OC40YzI5LDAsNTAuOS0xNC4zLDYyLjItMzAuMmgxLjNsMi45LDI1LjZIMTYwNC44eiBNMTU2NS43LDI1Ny43IGMwLDMuOC0wLjgsOC0yLjEsMTEuOGMtNS45LDE3LjItMjIuNywzNC00OS4yLDM0Yy0xOC45LDAtMzQuOS0xMS4zLTM0LjktMzUuM2MwLTM5LjUsNDUuOC00Ni42LDg2LjItNDUuOFYyNTcuN3ogTTE2OTguNSwzMjYuMiBsMS43LTMzLjZoMS4zYzE1LjEsMjYuOSwzOC43LDM4LjIsNjguMSwzOC4yYzQ1LjQsMCw5MS4yLTM2LjEsOTEuMi0xMDguOGMwLjQtNjEuNy0zNS4zLTEwMy43LTg1LjctMTAzLjcgYy0zMi44LDAtNTYuMywxNC43LTY5LjMsMzcuNGgtMC44VjI4aC0zNi42djI0NS43YzAsMTguMS0wLjgsMzguNi0xLjcsNTIuNUgxNjk4LjV6IE0xNzA0LjgsMjA4LjJjMC01LjksMS4zLTEwLjksMi4xLTE1LjEgYzcuNi0yOC4xLDMxLjEtNDUuNCw1Ni4zLTQ1LjRjMzkuNSwwLDYwLjUsMzQuOSw2MC41LDc1LjZjMCw0Ni42LTIzLjEsNzguMS02MS44LDc4LjFjLTI2LjksMC00OC4zLTE3LjYtNTUuNS00My4zIGMtMC44LTQuMi0xLjctOC44LTEuNy0xMy40VjIwOC4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzYxNjE2MSIgZD0iTTE1IDlIOXY2aDZWOXptLTIgNGgtMnYtMmgydjJ6bTgtMlY5aC0yVjdjMC0xLjEtLjktMi0yLTJoLTJWM2gtMnYyaC0yVjNIOXYySDdjLTEuMSAwLTIgLjktMiAydjJIM3YyaDJ2MkgzdjJoMnYyYzAgMS4xLjkgMiAyIDJoMnYyaDJ2LTJoMnYyaDJ2LTJoMmMxLjEgMCAyLS45IDItMnYtMmgydi0yaC0ydi0yaDJ6bS00IDZIN1Y3aDEwdjEweiIvPgo8L3N2Zz4K);
  --jp-icon-keyboard: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMTdjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0tOSAzaDJ2MmgtMlY4em0wIDNoMnYyaC0ydi0yek04IDhoMnYySDhWOHptMCAzaDJ2Mkg4di0yem0tMSAySDV2LTJoMnYyem0wLTNINVY4aDJ2MnptOSA3SDh2LTJoOHYyem0wLTRoLTJ2LTJoMnYyem0wLTNoLTJWOGgydjJ6bTMgM2gtMnYtMmgydjJ6bTAtM2gtMlY4aDJ2MnoiLz4KPC9zdmc+Cg==);
  --jp-icon-launch: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMzIgMzIiIHdpZHRoPSIzMiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0yNiwyOEg2YTIuMDAyNywyLjAwMjcsMCwwLDEtMi0yVjZBMi4wMDI3LDIuMDAyNywwLDAsMSw2LDRIMTZWNkg2VjI2SDI2VjE2aDJWMjZBMi4wMDI3LDIuMDAyNywwLDAsMSwyNiwyOFoiLz4KICAgIDxwb2x5Z29uIHBvaW50cz0iMjAgMiAyMCA0IDI2LjU4NiA0IDE4IDEyLjU4NiAxOS40MTQgMTQgMjggNS40MTQgMjggMTIgMzAgMTIgMzAgMiAyMCAyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-launcher: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkgMTlINVY1aDdWM0g1YTIgMiAwIDAwLTIgMnYxNGEyIDIgMCAwMDIgMmgxNGMxLjEgMCAyLS45IDItMnYtN2gtMnY3ek0xNCAzdjJoMy41OWwtOS44MyA5LjgzIDEuNDEgMS40MUwxOSA2LjQxVjEwaDJWM2gtN3oiLz4KPC9zdmc+Cg==);
  --jp-icon-line-form: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGZpbGw9IndoaXRlIiBkPSJNNS44OCA0LjEyTDEzLjc2IDEybC03Ljg4IDcuODhMOCAyMmwxMC0xMEw4IDJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-link: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMuOSAxMmMwLTEuNzEgMS4zOS0zLjEgMy4xLTMuMWg0VjdIN2MtMi43NiAwLTUgMi4yNC01IDVzMi4yNCA1IDUgNWg0di0xLjlIN2MtMS43MSAwLTMuMS0xLjM5LTMuMS0zLjF6TTggMTNoOHYtMkg4djJ6bTktNmgtNHYxLjloNGMxLjcxIDAgMy4xIDEuMzkgMy4xIDMuMXMtMS4zOSAzLjEtMy4xIDMuMWgtNFYxN2g0YzIuNzYgMCA1LTIuMjQgNS01cy0yLjI0LTUtNS01eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xOSA1djE0SDVWNWgxNG0xLjEtMkgzLjljLS41IDAtLjkuNC0uOS45djE2LjJjMCAuNC40LjkuOS45aDE2LjJjLjQgMCAuOS0uNS45LS45VjMuOWMwLS41LS41LS45LS45LS45ek0xMSA3aDZ2MmgtNlY3em0wIDRoNnYyaC02di0yem0wIDRoNnYyaC02ek03IDdoMnYySDd6bTAgNGgydjJIN3ptMCA0aDJ2Mkg3eiIvPgo8L3N2Zz4K);
  --jp-icon-markdown: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjN0IxRkEyIiBkPSJNNSAxNC45aDEybC02LjEgNnptOS40LTYuOGMwLTEuMy0uMS0yLjktLjEtNC41LS40IDEuNC0uOSAyLjktMS4zIDQuM2wtMS4zIDQuM2gtMkw4LjUgNy45Yy0uNC0xLjMtLjctMi45LTEtNC4zLS4xIDEuNi0uMSAzLjItLjIgNC42TDcgMTIuNEg0LjhsLjctMTFoMy4zTDEwIDVjLjQgMS4yLjcgMi43IDEgMy45LjMtMS4yLjctMi42IDEtMy45bDEuMi0zLjdoMy4zbC42IDExaC0yLjRsLS4zLTQuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-move-down: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNMTIuNDcxIDcuNTI4OTlDMTIuNzYzMiA3LjIzNjg0IDEyLjc2MzIgNi43NjMxNiAxMi40NzEgNi40NzEwMVY2LjQ3MTAxQzEyLjE3OSA2LjE3OTA1IDExLjcwNTcgNi4xNzg4NCAxMS40MTM1IDYuNDcwNTRMNy43NSAxMC4xMjc1VjEuNzVDNy43NSAxLjMzNTc5IDcuNDE0MjEgMSA3IDFWMUM2LjU4NTc5IDEgNi4yNSAxLjMzNTc5IDYuMjUgMS43NVYxMC4xMjc1TDIuNTk3MjYgNi40NjgyMkMyLjMwMzM4IDYuMTczODEgMS44MjY0MSA2LjE3MzU5IDEuNTMyMjYgNi40Njc3NFY2LjQ2Nzc0QzEuMjM4MyA2Ljc2MTcgMS4yMzgzIDcuMjM4MyAxLjUzMjI2IDcuNTMyMjZMNi4yOTI4OSAxMi4yOTI5QzYuNjgzNDIgMTIuNjgzNCA3LjMxNjU4IDEyLjY4MzQgNy43MDcxMSAxMi4yOTI5TDEyLjQ3MSA3LjUyODk5WiIgZmlsbD0iIzYxNjE2MSIvPgo8L3N2Zz4K);
  --jp-icon-move-up: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNMS41Mjg5OSA2LjQ3MTAxQzEuMjM2ODQgNi43NjMxNiAxLjIzNjg0IDcuMjM2ODQgMS41Mjg5OSA3LjUyODk5VjcuNTI4OTlDMS44MjA5NSA3LjgyMDk1IDIuMjk0MjYgNy44MjExNiAyLjU4NjQ5IDcuNTI5NDZMNi4yNSAzLjg3MjVWMTIuMjVDNi4yNSAxMi42NjQyIDYuNTg1NzkgMTMgNyAxM1YxM0M3LjQxNDIxIDEzIDcuNzUgMTIuNjY0MiA3Ljc1IDEyLjI1VjMuODcyNUwxMS40MDI3IDcuNTMxNzhDMTEuNjk2NiA3LjgyNjE5IDEyLjE3MzYgNy44MjY0MSAxMi40Njc3IDcuNTMyMjZWNy41MzIyNkMxMi43NjE3IDcuMjM4MyAxMi43NjE3IDYuNzYxNyAxMi40Njc3IDYuNDY3NzRMNy43MDcxMSAxLjcwNzExQzcuMzE2NTggMS4zMTY1OCA2LjY4MzQyIDEuMzE2NTggNi4yOTI4OSAxLjcwNzExTDEuNTI4OTkgNi40NzEwMVoiIGZpbGw9IiM2MTYxNjEiLz4KPC9zdmc+Cg==);
  --jp-icon-new-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwIDZoLThsLTItMkg0Yy0xLjExIDAtMS45OS44OS0xLjk5IDJMMiAxOGMwIDEuMTEuODkgMiAyIDJoMTZjMS4xMSAwIDItLjg5IDItMlY4YzAtMS4xMS0uODktMi0yLTJ6bS0xIDhoLTN2M2gtMnYtM2gtM3YtMmgzVjloMnYzaDN2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-not-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI1IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMTkgMTcuMTg0NCAyLjk2OTY4IDE0LjMwMzIgMS44NjA5NCAxMS40NDA5WiIvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24yIiBzdHJva2U9IiMzMzMzMzMiIHN0cm9rZS13aWR0aD0iMiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOS4zMTU5MiA5LjMyMDMxKSIgZD0iTTcuMzY4NDIgMEwwIDcuMzY0NzkiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDkuMzE1OTIgMTYuNjgzNikgc2NhbGUoMSAtMSkiIGQ9Ik03LjM2ODQyIDBMMCA3LjM2NDc5Ii8+Cjwvc3ZnPgo=);
  --jp-icon-notebook: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtbm90ZWJvb2staWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNFRjZDMDAiPgogICAgPHBhdGggZD0iTTE4LjcgMy4zdjE1LjRIMy4zVjMuM2gxNS40bTEuNS0xLjVIMS44djE4LjNoMTguM2wuMS0xOC4zeiIvPgogICAgPHBhdGggZD0iTTE2LjUgMTYuNWwtNS40LTQuMy01LjYgNC4zdi0xMWgxMXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-numbering: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTQgMTlINlYxOS41SDVWMjAuNUg2VjIxSDRWMjJIN1YxOEg0VjE5Wk01IDEwSDZWNkg0VjdINVYxMFpNNCAxM0g1LjhMNCAxNS4xVjE2SDdWMTVINS4yTDcgMTIuOVYxMkg0VjEzWk05IDdWOUgyM1Y3SDlaTTkgMjFIMjNWMTlIOVYyMVpNOSAxNUgyM1YxM0g5VjE1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-offline-bolt: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDIuMDJjLTUuNTEgMC05Ljk4IDQuNDctOS45OCA5Ljk4czQuNDcgOS45OCA5Ljk4IDkuOTggOS45OC00LjQ3IDkuOTgtOS45OFMxNy41MSAyLjAyIDEyIDIuMDJ6TTExLjQ4IDIwdi02LjI2SDhMMTMgNHY2LjI2aDMuMzVMMTEuNDggMjB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-palette: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE4IDEzVjIwSDRWNkg5LjAyQzkuMDcgNS4yOSA5LjI0IDQuNjIgOS41IDRINEMyLjkgNCAyIDQuOSAyIDZWMjBDMiAyMS4xIDIuOSAyMiA0IDIySDE4QzE5LjEgMjIgMjAgMjEuMSAyMCAyMFYxNUwxOCAxM1pNMTkuMyA4Ljg5QzE5Ljc0IDguMTkgMjAgNy4zOCAyMCA2LjVDMjAgNC4wMSAxNy45OSAyIDE1LjUgMkMxMy4wMSAyIDExIDQuMDEgMTEgNi41QzExIDguOTkgMTMuMDEgMTEgMTUuNDkgMTFDMTYuMzcgMTEgMTcuMTkgMTAuNzQgMTcuODggMTAuM0wyMSAxMy40MkwyMi40MiAxMkwxOS4zIDguODlaTTE1LjUgOUMxNC4xMiA5IDEzIDcuODggMTMgNi41QzEzIDUuMTIgMTQuMTIgNCAxNS41IDRDMTYuODggNCAxOCA1LjEyIDE4IDYuNUMxOCA3Ljg4IDE2Ljg4IDkgMTUuNSA5WiIvPgogICAgPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik00IDZIOS4wMTg5NEM5LjAwNjM5IDYuMTY1MDIgOSA2LjMzMTc2IDkgNi41QzkgOC44MTU3NyAxMC4yMTEgMTAuODQ4NyAxMi4wMzQzIDEySDlWMTRIMTZWMTIuOTgxMUMxNi41NzAzIDEyLjkzNzcgMTcuMTIgMTIuODIwNyAxNy42Mzk2IDEyLjYzOTZMMTggMTNWMjBINFY2Wk04IDhINlYxMEg4VjhaTTYgMTJIOFYxNEg2VjEyWk04IDE2SDZWMThIOFYxNlpNOSAxNkgxNlYxOEg5VjE2WiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-paste: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE5IDJoLTQuMThDMTQuNC44NCAxMy4zIDAgMTIgMGMtMS4zIDAtMi40Ljg0LTIuODIgMkg1Yy0xLjEgMC0yIC45LTIgMnYxNmMwIDEuMS45IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjRjMC0xLjEtLjktMi0yLTJ6bS03IDBjLjU1IDAgMSAuNDUgMSAxcy0uNDUgMS0xIDEtMS0uNDUtMS0xIC40NS0xIDEtMXptNyAxOEg1VjRoMnYzaDEwVjRoMnYxNnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-pdf: url(data:image/svg+xml;base64,PHN2ZwogICB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMiAyMiIgd2lkdGg9IjE2Ij4KICAgIDxwYXRoIHRyYW5zZm9ybT0icm90YXRlKDQ1KSIgY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI0ZGMkEyQSIKICAgICAgIGQ9Im0gMjIuMzQ0MzY5LC0zLjAxNjM2NDIgaCA1LjYzODYwNCB2IDEuNTc5MjQzMyBoIC0zLjU0OTIyNyB2IDEuNTA4NjkyOTkgaCAzLjMzNzU3NiBWIDEuNjUwODE1NCBoIC0zLjMzNzU3NiB2IDMuNDM1MjYxMyBoIC0yLjA4OTM3NyB6IG0gLTcuMTM2NDQ0LDEuNTc5MjQzMyB2IDQuOTQzOTU0MyBoIDAuNzQ4OTIgcSAxLjI4MDc2MSwwIDEuOTUzNzAzLC0wLjYzNDk1MzUgMC42NzgzNjksLTAuNjM0OTUzNSAwLjY3ODM2OSwtMS44NDUxNjQxIDAsLTEuMjA0NzgzNTUgLTAuNjcyOTQyLC0xLjgzNDMxMDExIC0wLjY3Mjk0MiwtMC42Mjk1MjY1OSAtMS45NTkxMywtMC42Mjk1MjY1OSB6IG0gLTIuMDg5Mzc3LC0xLjU3OTI0MzMgaCAyLjIwMzM0MyBxIDEuODQ1MTY0LDAgMi43NDYwMzksMC4yNjU5MjA3IDAuOTA2MzAxLDAuMjYwNDkzNyAxLjU1MjEwOCwwLjg5MDAyMDMgMC41Njk4MywwLjU0ODEyMjMgMC44NDY2MDUsMS4yNjQ0ODAwNiAwLjI3Njc3NCwwLjcxNjM1NzgxIDAuMjc2Nzc0LDEuNjIyNjU4OTQgMCwwLjkxNzE1NTEgLTAuMjc2Nzc0LDEuNjM4OTM5OSAtMC4yNzY3NzUsMC43MTYzNTc4IC0wLjg0NjYwNSwxLjI2NDQ4IC0wLjY1MTIzNCwwLjYyOTUyNjYgLTEuNTYyOTYyLDAuODk1NDQ3MyAtMC45MTE3MjgsMC4yNjA0OTM3IC0yLjczNTE4NSwwLjI2MDQ5MzcgaCAtMi4yMDMzNDMgeiBtIC04LjE0NTg1NjUsMCBoIDMuNDY3ODIzIHEgMS41NDY2ODE2LDAgMi4zNzE1Nzg1LDAuNjg5MjIzIDAuODMwMzI0LDAuNjgzNzk2MSAwLjgzMDMyNCwxLjk1MzcwMzE0IDAsMS4yNzUzMzM5NyAtMC44MzAzMjQsMS45NjQ1NTcwNiBRIDkuOTg3MTk2MSwyLjI3NDkxNSA4LjQ0MDUxNDUsMi4yNzQ5MTUgSCA3LjA2MjA2ODQgViA1LjA4NjA3NjcgSCA0Ljk3MjY5MTUgWiBtIDIuMDg5Mzc2OSwxLjUxNDExOTkgdiAyLjI2MzAzOTQzIGggMS4xNTU5NDEgcSAwLjYwNzgxODgsMCAwLjkzODg2MjksLTAuMjkzMDU1NDcgMC4zMzEwNDQxLC0wLjI5ODQ4MjQxIDAuMzMxMDQ0MSwtMC44NDExNzc3MiAwLC0wLjU0MjY5NTMxIC0wLjMzMTA0NDEsLTAuODM1NzUwNzQgLTAuMzMxMDQ0MSwtMC4yOTMwNTU1IC0wLjkzODg2MjksLTAuMjkzMDU1NSB6IgovPgo8L3N2Zz4K);
  --jp-icon-python: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iLTEwIC0xMCAxMzEuMTYxMzYxNjk0MzM1OTQgMTMyLjM4ODk5OTkzODk2NDg0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMzA2OTk4IiBkPSJNIDU0LjkxODc4NSw5LjE5Mjc0MjFlLTQgQyA1MC4zMzUxMzIsMC4wMjIyMTcyNyA0NS45NTc4NDYsMC40MTMxMzY5NyA0Mi4xMDYyODUsMS4wOTQ2NjkzIDMwLjc2MDA2OSwzLjA5OTE3MzEgMjguNzAwMDM2LDcuMjk0NzcxNCAyOC43MDAwMzUsMTUuMDMyMTY5IHYgMTAuMjE4NzUgaCAyNi44MTI1IHYgMy40MDYyNSBoIC0yNi44MTI1IC0xMC4wNjI1IGMgLTcuNzkyNDU5LDAgLTE0LjYxNTc1ODgsNC42ODM3MTcgLTE2Ljc0OTk5OTgsMTMuNTkzNzUgLTIuNDYxODE5OTgsMTAuMjEyOTY2IC0yLjU3MTAxNTA4LDE2LjU4NjAyMyAwLDI3LjI1IDEuOTA1OTI4Myw3LjkzNzg1MiA2LjQ1NzU0MzIsMTMuNTkzNzQ4IDE0LjI0OTk5OTgsMTMuNTkzNzUgaCA5LjIxODc1IHYgLTEyLjI1IGMgMCwtOC44NDk5MDIgNy42NTcxNDQsLTE2LjY1NjI0OCAxNi43NSwtMTYuNjU2MjUgaCAyNi43ODEyNSBjIDcuNDU0OTUxLDAgMTMuNDA2MjUzLC02LjEzODE2NCAxMy40MDYyNSwtMTMuNjI1IHYgLTI1LjUzMTI1IGMgMCwtNy4yNjYzMzg2IC02LjEyOTk4LC0xMi43MjQ3NzcxIC0xMy40MDYyNSwtMTMuOTM3NDk5NyBDIDY0LjI4MTU0OCwwLjMyNzk0Mzk3IDU5LjUwMjQzOCwtMC4wMjAzNzkwMyA1NC45MTg3ODUsOS4xOTI3NDIxZS00IFogbSAtMTQuNSw4LjIxODc1MDEyNTc5IGMgMi43Njk1NDcsMCA1LjAzMTI1LDIuMjk4NjQ1NiA1LjAzMTI1LDUuMTI0OTk5NiAtMmUtNiwyLjgxNjMzNiAtMi4yNjE3MDMsNS4wOTM3NSAtNS4wMzEyNSw1LjA5Mzc1IC0yLjc3OTQ3NiwtMWUtNiAtNS4wMzEyNSwtMi4yNzc0MTUgLTUuMDMxMjUsLTUuMDkzNzUgLTEwZS03LC0yLjgyNjM1MyAyLjI1MTc3NCwtNS4xMjQ5OTk2IDUuMDMxMjUsLTUuMTI0OTk5NiB6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2ZmZDQzYiIgZD0ibSA4NS42Mzc1MzUsMjguNjU3MTY5IHYgMTEuOTA2MjUgYyAwLDkuMjMwNzU1IC03LjgyNTg5NSwxNi45OTk5OTkgLTE2Ljc1LDE3IGggLTI2Ljc4MTI1IGMgLTcuMzM1ODMzLDAgLTEzLjQwNjI0OSw2LjI3ODQ4MyAtMTMuNDA2MjUsMTMuNjI1IHYgMjUuNTMxMjQ3IGMgMCw3LjI2NjM0NCA2LjMxODU4OCwxMS41NDAzMjQgMTMuNDA2MjUsMTMuNjI1MDA0IDguNDg3MzMxLDIuNDk1NjEgMTYuNjI2MjM3LDIuOTQ2NjMgMjYuNzgxMjUsMCA2Ljc1MDE1NSwtMS45NTQzOSAxMy40MDYyNTMsLTUuODg3NjEgMTMuNDA2MjUsLTEzLjYyNTAwNCBWIDg2LjUwMDkxOSBoIC0yNi43ODEyNSB2IC0zLjQwNjI1IGggMjYuNzgxMjUgMTMuNDA2MjU0IGMgNy43OTI0NjEsMCAxMC42OTYyNTEsLTUuNDM1NDA4IDEzLjQwNjI0MSwtMTMuNTkzNzUgMi43OTkzMywtOC4zOTg4ODYgMi42ODAyMiwtMTYuNDc1Nzc2IDAsLTI3LjI1IC0xLjkyNTc4LC03Ljc1NzQ0MSAtNS42MDM4NywtMTMuNTkzNzUgLTEzLjQwNjI0MSwtMTMuNTkzNzUgeiBtIC0xNS4wNjI1LDY0LjY1NjI1IGMgMi43Nzk0NzgsM2UtNiA1LjAzMTI1LDIuMjc3NDE3IDUuMDMxMjUsNS4wOTM3NDcgLTJlLTYsMi44MjYzNTQgLTIuMjUxNzc1LDUuMTI1MDA0IC01LjAzMTI1LDUuMTI1MDA0IC0yLjc2OTU1LDAgLTUuMDMxMjUsLTIuMjk4NjUgLTUuMDMxMjUsLTUuMTI1MDA0IDJlLTYsLTIuODE2MzMgMi4yNjE2OTcsLTUuMDkzNzQ3IDUuMDMxMjUsLTUuMDkzNzQ3IHoiLz4KPC9zdmc+Cg==);
  --jp-icon-r-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjE5NkYzIiBkPSJNNC40IDIuNWMxLjItLjEgMi45LS4zIDQuOS0uMyAyLjUgMCA0LjEuNCA1LjIgMS4zIDEgLjcgMS41IDEuOSAxLjUgMy41IDAgMi0xLjQgMy41LTIuOSA0LjEgMS4yLjQgMS43IDEuNiAyLjIgMyAuNiAxLjkgMSAzLjkgMS4zIDQuNmgtMy44Yy0uMy0uNC0uOC0xLjctMS4yLTMuN3MtMS4yLTIuNi0yLjYtMi42aC0uOXY2LjRINC40VjIuNXptMy43IDYuOWgxLjRjMS45IDAgMi45LS45IDIuOS0yLjNzLTEtMi4zLTIuOC0yLjNjLS43IDAtMS4zIDAtMS42LjJ2NC41aC4xdi0uMXoiLz4KPC9zdmc+Cg==);
  --jp-icon-react: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMTUwIDE1MCA1NDEuOSAyOTUuMyI+CiAgPGcgY2xhc3M9ImpwLWljb24tYnJhbmQyIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxREFGQiI+CiAgICA8cGF0aCBkPSJNNjY2LjMgMjk2LjVjMC0zMi41LTQwLjctNjMuMy0xMDMuMS04Mi40IDE0LjQtNjMuNiA4LTExNC4yLTIwLjItMTMwLjQtNi41LTMuOC0xNC4xLTUuNi0yMi40LTUuNnYyMi4zYzQuNiAwIDguMy45IDExLjQgMi42IDEzLjYgNy44IDE5LjUgMzcuNSAxNC45IDc1LjctMS4xIDkuNC0yLjkgMTkuMy01LjEgMjkuNC0xOS42LTQuOC00MS04LjUtNjMuNS0xMC45LTEzLjUtMTguNS0yNy41LTM1LjMtNDEuNi01MCAzMi42LTMwLjMgNjMuMi00Ni45IDg0LTQ2LjlWNzhjLTI3LjUgMC02My41IDE5LjYtOTkuOSA1My42LTM2LjQtMzMuOC03Mi40LTUzLjItOTkuOS01My4ydjIyLjNjMjAuNyAwIDUxLjQgMTYuNSA4NCA0Ni42LTE0IDE0LjctMjggMzEuNC00MS4zIDQ5LjktMjIuNiAyLjQtNDQgNi4xLTYzLjYgMTEtMi4zLTEwLTQtMTkuNy01LjItMjktNC43LTM4LjIgMS4xLTY3LjkgMTQuNi03NS44IDMtMS44IDYuOS0yLjYgMTEuNS0yLjZWNzguNWMtOC40IDAtMTYgMS44LTIyLjYgNS42LTI4LjEgMTYuMi0zNC40IDY2LjctMTkuOSAxMzAuMS02Mi4yIDE5LjItMTAyLjcgNDkuOS0xMDIuNyA4Mi4zIDAgMzIuNSA0MC43IDYzLjMgMTAzLjEgODIuNC0xNC40IDYzLjYtOCAxMTQuMiAyMC4yIDEzMC40IDYuNSAzLjggMTQuMSA1LjYgMjIuNSA1LjYgMjcuNSAwIDYzLjUtMTkuNiA5OS45LTUzLjYgMzYuNCAzMy44IDcyLjQgNTMuMiA5OS45IDUzLjIgOC40IDAgMTYtMS44IDIyLjYtNS42IDI4LjEtMTYuMiAzNC40LTY2LjcgMTkuOS0xMzAuMSA2Mi0xOS4xIDEwMi41LTQ5LjkgMTAyLjUtODIuM3ptLTEzMC4yLTY2LjdjLTMuNyAxMi45LTguMyAyNi4yLTEzLjUgMzkuNS00LjEtOC04LjQtMTYtMTMuMS0yNC00LjYtOC05LjUtMTUuOC0xNC40LTIzLjQgMTQuMiAyLjEgMjcuOSA0LjcgNDEgNy45em0tNDUuOCAxMDYuNWMtNy44IDEzLjUtMTUuOCAyNi4zLTI0LjEgMzguMi0xNC45IDEuMy0zMCAyLTQ1LjIgMi0xNS4xIDAtMzAuMi0uNy00NS0xLjktOC4zLTExLjktMTYuNC0yNC42LTI0LjItMzgtNy42LTEzLjEtMTQuNS0yNi40LTIwLjgtMzkuOCA2LjItMTMuNCAxMy4yLTI2LjggMjAuNy0zOS45IDcuOC0xMy41IDE1LjgtMjYuMyAyNC4xLTM4LjIgMTQuOS0xLjMgMzAtMiA0NS4yLTIgMTUuMSAwIDMwLjIuNyA0NSAxLjkgOC4zIDExLjkgMTYuNCAyNC42IDI0LjIgMzggNy42IDEzLjEgMTQuNSAyNi40IDIwLjggMzkuOC02LjMgMTMuNC0xMy4yIDI2LjgtMjAuNyAzOS45em0zMi4zLTEzYzUuNCAxMy40IDEwIDI2LjggMTMuOCAzOS44LTEzLjEgMy4yLTI2LjkgNS45LTQxLjIgOCA0LjktNy43IDkuOC0xNS42IDE0LjQtMjMuNyA0LjYtOCA4LjktMTYuMSAxMy0yNC4xek00MjEuMiA0MzBjLTkuMy05LjYtMTguNi0yMC4zLTI3LjgtMzIgOSAuNCAxOC4yLjcgMjcuNS43IDkuNCAwIDE4LjctLjIgMjcuOC0uNy05IDExLjctMTguMyAyMi40LTI3LjUgMzJ6bS03NC40LTU4LjljLTE0LjItMi4xLTI3LjktNC43LTQxLTcuOSAzLjctMTIuOSA4LjMtMjYuMiAxMy41LTM5LjUgNC4xIDggOC40IDE2IDEzLjEgMjQgNC43IDggOS41IDE1LjggMTQuNCAyMy40ek00MjAuNyAxNjNjOS4zIDkuNiAxOC42IDIwLjMgMjcuOCAzMi05LS40LTE4LjItLjctMjcuNS0uNy05LjQgMC0xOC43LjItMjcuOC43IDktMTEuNyAxOC4zLTIyLjQgMjcuNS0zMnptLTc0IDU4LjljLTQuOSA3LjctOS44IDE1LjYtMTQuNCAyMy43LTQuNiA4LTguOSAxNi0xMyAyNC01LjQtMTMuNC0xMC0yNi44LTEzLjgtMzkuOCAxMy4xLTMuMSAyNi45LTUuOCA0MS4yLTcuOXptLTkwLjUgMTI1LjJjLTM1LjQtMTUuMS01OC4zLTM0LjktNTguMy01MC42IDAtMTUuNyAyMi45LTM1LjYgNTguMy01MC42IDguNi0zLjcgMTgtNyAyNy43LTEwLjEgNS43IDE5LjYgMTMuMiA0MCAyMi41IDYwLjktOS4yIDIwLjgtMTYuNiA0MS4xLTIyLjIgNjAuNi05LjktMy4xLTE5LjMtNi41LTI4LTEwLjJ6TTMxMCA0OTBjLTEzLjYtNy44LTE5LjUtMzcuNS0xNC45LTc1LjcgMS4xLTkuNCAyLjktMTkuMyA1LjEtMjkuNCAxOS42IDQuOCA0MSA4LjUgNjMuNSAxMC45IDEzLjUgMTguNSAyNy41IDM1LjMgNDEuNiA1MC0zMi42IDMwLjMtNjMuMiA0Ni45LTg0IDQ2LjktNC41LS4xLTguMy0xLTExLjMtMi43em0yMzcuMi03Ni4yYzQuNyAzOC4yLTEuMSA2Ny45LTE0LjYgNzUuOC0zIDEuOC02LjkgMi42LTExLjUgMi42LTIwLjcgMC01MS40LTE2LjUtODQtNDYuNiAxNC0xNC43IDI4LTMxLjQgNDEuMy00OS45IDIyLjYtMi40IDQ0LTYuMSA2My42LTExIDIuMyAxMC4xIDQuMSAxOS44IDUuMiAyOS4xem0zOC41LTY2LjdjLTguNiAzLjctMTggNy0yNy43IDEwLjEtNS43LTE5LjYtMTMuMi00MC0yMi41LTYwLjkgOS4yLTIwLjggMTYuNi00MS4xIDIyLjItNjAuNiA5LjkgMy4xIDE5LjMgNi41IDI4LjEgMTAuMiAzNS40IDE1LjEgNTguMyAzNC45IDU4LjMgNTAuNi0uMSAxNS43LTIzIDM1LjYtNTguNCA1MC42ek0zMjAuOCA3OC40eiIvPgogICAgPGNpcmNsZSBjeD0iNDIwLjkiIGN5PSIyOTYuNSIgcj0iNDUuNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-redo: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCBkPSJNMCAwaDI0djI0SDB6IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTE4LjQgMTAuNkMxNi41NSA4Ljk5IDE0LjE1IDggMTEuNSA4Yy00LjY1IDAtOC41OCAzLjAzLTkuOTYgNy4yMkwzLjkgMTZjMS4wNS0zLjE5IDQuMDUtNS41IDcuNi01LjUgMS45NSAwIDMuNzMuNzIgNS4xMiAxLjg4TDEzIDE2aDlWN2wtMy42IDMuNnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-refresh: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTkgMTMuNWMtMi40OSAwLTQuNS0yLjAxLTQuNS00LjVTNi41MSA0LjUgOSA0LjVjMS4yNCAwIDIuMzYuNTIgMy4xNyAxLjMzTDEwIDhoNVYzbC0xLjc2IDEuNzZDMTIuMTUgMy42OCAxMC42NiAzIDkgMyA1LjY5IDMgMy4wMSA1LjY5IDMuMDEgOVM1LjY5IDE1IDkgMTVjMi45NyAwIDUuNDMtMi4xNiA1LjktNWgtMS41MmMtLjQ2IDItMi4yNCAzLjUtNC4zOCAzLjV6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-regex: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiBmaWxsPSIjRkZGIj4KICAgIDxjaXJjbGUgY2xhc3M9InN0MiIgY3g9IjUuNSIgY3k9IjE0LjUiIHI9IjEuNSIvPgogICAgPHJlY3QgeD0iMTIiIHk9IjQiIGNsYXNzPSJzdDIiIHdpZHRoPSIxIiBoZWlnaHQ9IjgiLz4KICAgIDxyZWN0IHg9IjguNSIgeT0iNy41IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjg2NiAtMC41IDAuNSAwLjg2NiAtMi4zMjU1IDcuMzIxOSkiIGNsYXNzPSJzdDIiIHdpZHRoPSI4IiBoZWlnaHQ9IjEiLz4KICAgIDxyZWN0IHg9IjEyIiB5PSI0IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjUgLTAuODY2IDAuODY2IDAuNSAtMC42Nzc5IDE0LjgyNTIpIiBjbGFzcz0ic3QyIiB3aWR0aD0iMSIgaGVpZ2h0PSI4Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-run: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTggNXYxNGwxMS03eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-running: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMjU2IDhDMTE5IDggOCAxMTkgOCAyNTZzMTExIDI0OCAyNDggMjQ4IDI0OC0xMTEgMjQ4LTI0OFMzOTMgOCAyNTYgOHptOTYgMzI4YzAgOC44LTcuMiAxNi0xNiAxNkgxNzZjLTguOCAwLTE2LTcuMi0xNi0xNlYxNzZjMC04LjggNy4yLTE2IDE2LTE2aDE2MGM4LjggMCAxNiA3LjIgMTYgMTZ2MTYweiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-save: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE3IDNINWMtMS4xMSAwLTIgLjktMiAydjE0YzAgMS4xLjg5IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjdsLTQtNHptLTUgMTZjLTEuNjYgMC0zLTEuMzQtMy0zczEuMzQtMyAzLTMgMyAxLjM0IDMgMy0xLjM0IDMtMyAzem0zLTEwSDVWNWgxMHY0eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-search: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjEsMTAuOWgtMC43bC0wLjItMC4yYzAuOC0wLjksMS4zLTIuMiwxLjMtMy41YzAtMy0yLjQtNS40LTUuNC01LjRTMS44LDQuMiwxLjgsNy4xczIuNCw1LjQsNS40LDUuNCBjMS4zLDAsMi41LTAuNSwzLjUtMS4zbDAuMiwwLjJ2MC43bDQuMSw0LjFsMS4yLTEuMkwxMi4xLDEwLjl6IE03LjEsMTAuOWMtMi4xLDAtMy43LTEuNy0zLjctMy43czEuNy0zLjcsMy43LTMuN3MzLjcsMS43LDMuNywzLjcgUzkuMiwxMC45LDcuMSwxMC45eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-settings: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuNDMgMTIuOThjLjA0LS4zMi4wNy0uNjQuMDctLjk4cy0uMDMtLjY2LS4wNy0uOThsMi4xMS0xLjY1Yy4xOS0uMTUuMjQtLjQyLjEyLS42NGwtMi0zLjQ2Yy0uMTItLjIyLS4zOS0uMy0uNjEtLjIybC0yLjQ5IDFjLS41Mi0uNC0xLjA4LS43My0xLjY5LS45OGwtLjM4LTIuNjVBLjQ4OC40ODggMCAwMDE0IDJoLTRjLS4yNSAwLS40Ni4xOC0uNDkuNDJsLS4zOCAyLjY1Yy0uNjEuMjUtMS4xNy41OS0xLjY5Ljk4bC0yLjQ5LTFjLS4yMy0uMDktLjQ5IDAtLjYxLjIybC0yIDMuNDZjLS4xMy4yMi0uMDcuNDkuMTIuNjRsMi4xMSAxLjY1Yy0uMDQuMzItLjA3LjY1LS4wNy45OHMuMDMuNjYuMDcuOThsLTIuMTEgMS42NWMtLjE5LjE1LS4yNC40Mi0uMTIuNjRsMiAzLjQ2Yy4xMi4yMi4zOS4zLjYxLjIybDIuNDktMWMuNTIuNCAxLjA4LjczIDEuNjkuOThsLjM4IDIuNjVjLjAzLjI0LjI0LjQyLjQ5LjQyaDRjLjI1IDAgLjQ2LS4xOC40OS0uNDJsLjM4LTIuNjVjLjYxLS4yNSAxLjE3LS41OSAxLjY5LS45OGwyLjQ5IDFjLjIzLjA5LjQ5IDAgLjYxLS4yMmwyLTMuNDZjLjEyLS4yMi4wNy0uNDktLjEyLS42NGwtMi4xMS0xLjY1ek0xMiAxNS41Yy0xLjkzIDAtMy41LTEuNTctMy41LTMuNXMxLjU3LTMuNSAzLjUtMy41IDMuNSAxLjU3IDMuNSAzLjUtMS41NyAzLjUtMy41IDMuNXoiLz4KPC9zdmc+Cg==);
  --jp-icon-share: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTSAxOCAyIEMgMTYuMzU0OTkgMiAxNSAzLjM1NDk5MDQgMTUgNSBDIDE1IDUuMTkwOTUyOSAxNS4wMjE3OTEgNS4zNzcxMjI0IDE1LjA1NjY0MSA1LjU1ODU5MzggTCA3LjkyMTg3NSA5LjcyMDcwMzEgQyA3LjM5ODUzOTkgOS4yNzc4NTM5IDYuNzMyMDc3MSA5IDYgOSBDIDQuMzU0OTkwNCA5IDMgMTAuMzU0OTkgMyAxMiBDIDMgMTMuNjQ1MDEgNC4zNTQ5OTA0IDE1IDYgMTUgQyA2LjczMjA3NzEgMTUgNy4zOTg1Mzk5IDE0LjcyMjE0NiA3LjkyMTg3NSAxNC4yNzkyOTcgTCAxNS4wNTY2NDEgMTguNDM5NDUzIEMgMTUuMDIxNTU1IDE4LjYyMTUxNCAxNSAxOC44MDgzODYgMTUgMTkgQyAxNSAyMC42NDUwMSAxNi4zNTQ5OSAyMiAxOCAyMiBDIDE5LjY0NTAxIDIyIDIxIDIwLjY0NTAxIDIxIDE5IEMgMjEgMTcuMzU0OTkgMTkuNjQ1MDEgMTYgMTggMTYgQyAxNy4yNjc0OCAxNiAxNi42MDE1OTMgMTYuMjc5MzI4IDE2LjA3ODEyNSAxNi43MjI2NTYgTCA4Ljk0MzM1OTQgMTIuNTU4NTk0IEMgOC45NzgyMDk1IDEyLjM3NzEyMiA5IDEyLjE5MDk1MyA5IDEyIEMgOSAxMS44MDkwNDcgOC45NzgyMDk1IDExLjYyMjg3OCA4Ljk0MzM1OTQgMTEuNDQxNDA2IEwgMTYuMDc4MTI1IDcuMjc5Mjk2OSBDIDE2LjYwMTQ2IDcuNzIyMTQ2MSAxNy4yNjc5MjMgOCAxOCA4IEMgMTkuNjQ1MDEgOCAyMSA2LjY0NTAwOTYgMjEgNSBDIDIxIDMuMzU0OTkwNCAxOS42NDUwMSAyIDE4IDIgeiBNIDE4IDQgQyAxOC41NjQxMjkgNCAxOSA0LjQzNTg3MDYgMTkgNSBDIDE5IDUuNTY0MTI5NCAxOC41NjQxMjkgNiAxOCA2IEMgMTcuNDM1ODcxIDYgMTcgNS41NjQxMjk0IDE3IDUgQyAxNyA0LjQzNTg3MDYgMTcuNDM1ODcxIDQgMTggNCB6IE0gNiAxMSBDIDYuNTY0MTI5NCAxMSA3IDExLjQzNTg3MSA3IDEyIEMgNyAxMi41NjQxMjkgNi41NjQxMjk0IDEzIDYgMTMgQyA1LjQzNTg3MDYgMTMgNSAxMi41NjQxMjkgNSAxMiBDIDUgMTEuNDM1ODcxIDUuNDM1ODcwNiAxMSA2IDExIHogTSAxOCAxOCBDIDE4LjU2NDEyOSAxOCAxOSAxOC40MzU4NzEgMTkgMTkgQyAxOSAxOS41NjQxMjkgMTguNTY0MTI5IDIwIDE4IDIwIEMgMTcuNDM1ODcxIDIwIDE3IDE5LjU2NDEyOSAxNyAxOSBDIDE3IDE4LjQzNTg3MSAxNy40MzU4NzEgMTggMTggMTggeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-spreadsheet: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNENBRjUwIiBkPSJNMi4yIDIuMnYxNy42aDE3LjZWMi4ySDIuMnptMTUuNCA3LjdoLTUuNVY0LjRoNS41djUuNXpNOS45IDQuNHY1LjVINC40VjQuNGg1LjV6bS01LjUgNy43aDUuNXY1LjVINC40di01LjV6bTcuNyA1LjV2LTUuNWg1LjV2NS41aC01LjV6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-stop: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik02IDZoMTJ2MTJINnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tab: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIxIDNIM2MtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxOGMxLjEgMCAyLS45IDItMlY1YzAtMS4xLS45LTItMi0yem0wIDE2SDNWNWgxMHY0aDh2MTB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-table-rows: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMSw4SDNWNGgxOFY4eiBNMjEsMTBIM3Y0aDE4VjEweiBNMjEsMTZIM3Y0aDE4VjE2eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-tag: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCA0MyAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTI4LjgzMzIgMTIuMzM0TDMyLjk5OTggMTYuNTAwN0wzNy4xNjY1IDEyLjMzNEgyOC44MzMyWiIvPgoJCTxwYXRoIGQ9Ik0xNi4yMDk1IDIxLjYxMDRDMTUuNjg3MyAyMi4xMjk5IDE0Ljg0NDMgMjIuMTI5OSAxNC4zMjQ4IDIxLjYxMDRMNi45ODI5IDE0LjcyNDVDNi41NzI0IDE0LjMzOTQgNi4wODMxMyAxMy42MDk4IDYuMDQ3ODYgMTMuMDQ4MkM1Ljk1MzQ3IDExLjUyODggNi4wMjAwMiA4LjYxOTQ0IDYuMDY2MjEgNy4wNzY5NUM2LjA4MjgxIDYuNTE0NzcgNi41NTU0OCA2LjA0MzQ3IDcuMTE4MDQgNi4wMzA1NUM5LjA4ODYzIDUuOTg0NzMgMTMuMjYzOCA1LjkzNTc5IDEzLjY1MTggNi4zMjQyNUwyMS43MzY5IDEzLjYzOUMyMi4yNTYgMTQuMTU4NSAyMS43ODUxIDE1LjQ3MjQgMjEuMjYyIDE1Ljk5NDZMMTYuMjA5NSAyMS42MTA0Wk05Ljc3NTg1IDguMjY1QzkuMzM1NTEgNy44MjU2NiA4LjYyMzUxIDcuODI1NjYgOC4xODI4IDguMjY1QzcuNzQzNDYgOC43MDU3MSA3Ljc0MzQ2IDkuNDE3MzMgOC4xODI4IDkuODU2NjdDOC42MjM4MiAxMC4yOTY0IDkuMzM1ODIgMTAuMjk2NCA5Ljc3NTg1IDkuODU2NjdDMTAuMjE1NiA5LjQxNzMzIDEwLjIxNTYgOC43MDUzMyA5Ljc3NTg1IDguMjY1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-terminal: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiA+CiAgICA8cmVjdCBjbGFzcz0ianAtdGVybWluYWwtaWNvbi1iYWNrZ3JvdW5kLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZSIgd2lkdGg9IjIwIiBoZWlnaHQ9IjIwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyIDIpIiBmaWxsPSIjMzMzMzMzIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtdGVybWluYWwtaWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUtaW52ZXJzZSIgZD0iTTUuMDU2NjQgOC43NjE3MkM1LjA1NjY0IDguNTk3NjYgNS4wMzEyNSA4LjQ1MzEyIDQuOTgwNDcgOC4zMjgxMkM0LjkzMzU5IDguMTk5MjIgNC44NTU0NyA4LjA4MjAzIDQuNzQ2MDkgNy45NzY1NkM0LjY0MDYyIDcuODcxMDkgNC41IDcuNzc1MzkgNC4zMjQyMiA3LjY4OTQ1QzQuMTUyMzQgNy41OTk2MSAzLjk0MzM2IDcuNTExNzIgMy42OTcyNyA3LjQyNTc4QzMuMzAyNzMgNy4yODUxNiAyLjk0MzM2IDcuMTM2NzIgMi42MTkxNCA2Ljk4MDQ3QzIuMjk0OTIgNi44MjQyMiAyLjAxNzU4IDYuNjQyNTggMS43ODcxMSA2LjQzNTU1QzEuNTYwNTUgNi4yMjg1MiAxLjM4NDc3IDUuOTg4MjggMS4yNTk3NyA1LjcxNDg0QzEuMTM0NzcgNS40Mzc1IDEuMDcyMjcgNS4xMDkzOCAxLjA3MjI3IDQuNzMwNDdDMS4wNzIyNyA0LjM5ODQ0IDEuMTI4OTEgNC4wOTU3IDEuMjQyMTkgMy44MjIyN0MxLjM1NTQ3IDMuNTQ0OTIgMS41MTU2MiAzLjMwNDY5IDEuNzIyNjYgMy4xMDE1NkMxLjkyOTY5IDIuODk4NDQgMi4xNzk2OSAyLjczNDM3IDIuNDcyNjYgMi42MDkzOEMyLjc2NTYyIDIuNDg0MzggMy4wOTE4IDIuNDA0MyAzLjQ1MTE3IDIuMzY5MTRWMS4xMDkzOEg0LjM4ODY3VjIuMzgwODZDNC43NDAyMyAyLjQyNzczIDUuMDU2NjQgMi41MjM0NCA1LjMzNzg5IDIuNjY3OTdDNS42MTkxNCAyLjgxMjUgNS44NTc0MiAzLjAwMTk1IDYuMDUyNzMgMy4yMzYzM0M2LjI1MTk1IDMuNDY2OCA2LjQwNDMgMy43NDAyMyA2LjUwOTc3IDQuMDU2NjRDNi42MTkxNCA0LjM2OTE0IDYuNjczODMgNC43MjA3IDYuNjczODMgNS4xMTEzM0g1LjA0NDkyQzUuMDQ0OTIgNC42Mzg2NyA0LjkzNzUgNC4yODEyNSA0LjcyMjY2IDQuMDM5MDZDNC41MDc4MSAzLjc5Mjk3IDQuMjE2OCAzLjY2OTkyIDMuODQ5NjEgMy42Njk5MkMzLjY1MDM5IDMuNjY5OTIgMy40NzY1NiAzLjY5NzI3IDMuMzI4MTIgMy43NTE5NUMzLjE4MzU5IDMuODAyNzMgMy4wNjQ0NSAzLjg3Njk1IDIuOTcwNyAzLjk3NDYxQzIuODc2OTUgNC4wNjgzNiAyLjgwNjY0IDQuMTc5NjkgMi43NTk3NyA0LjMwODU5QzIuNzE2OCA0LjQzNzUgMi42OTUzMSA0LjU3ODEyIDIuNjk1MzEgNC43MzA0N0MyLjY5NTMxIDQuODgyODEgMi43MTY4IDUuMDE5NTMgMi43NTk3NyA1LjE0MDYyQzIuODA2NjQgNS4yNTc4MSAyLjg4MjgxIDUuMzY3MTkgMi45ODgyOCA1LjQ2ODc1QzMuMDk3NjYgNS41NzAzMSAzLjI0MDIzIDUuNjY3OTcgMy40MTYwMiA1Ljc2MTcyQzMuNTkxOCA1Ljg1MTU2IDMuODEwNTUgNS45NDMzNiA0LjA3MjI3IDYuMDM3MTFDNC40NjY4IDYuMTg1NTUgNC44MjQyMiA2LjMzOTg0IDUuMTQ0NTMgNi41QzUuNDY0ODQgNi42NTYyNSA1LjczODI4IDYuODM5ODQgNS45NjQ4NCA3LjA1MDc4QzYuMTk1MzEgNy4yNTc4MSA2LjM3MTA5IDcuNSA2LjQ5MjE5IDcuNzc3MzRDNi42MTcxOSA4LjA1MDc4IDYuNjc5NjkgOC4zNzUgNi42Nzk2OSA4Ljc1QzYuNjc5NjkgOS4wOTM3NSA2LjYyMzA1IDkuNDA0MyA2LjUwOTc3IDkuNjgxNjRDNi4zOTY0OCA5Ljk1NTA4IDYuMjM0MzggMTAuMTkxNCA2LjAyMzQ0IDEwLjM5MDZDNS44MTI1IDEwLjU4OTggNS41NTg1OSAxMC43NSA1LjI2MTcyIDEwLjg3MTFDNC45NjQ4NCAxMC45ODgzIDQuNjMyODEgMTEuMDY0NSA0LjI2NTYyIDExLjA5OTZWMTIuMjQ4SDMuMzMzOThWMTEuMDk5NkMzLjAwMTk1IDExLjA2ODQgMi42Nzk2OSAxMC45OTYxIDIuMzY3MTkgMTAuODgyOEMyLjA1NDY5IDEwLjc2NTYgMS43NzczNCAxMC41OTc3IDEuNTM1MTYgMTAuMzc4OUMxLjI5Njg4IDEwLjE2MDIgMS4xMDU0NyA5Ljg4NDc3IDAuOTYwOTM4IDkuNTUyNzNDMC44MTY0MDYgOS4yMTY4IDAuNzQ0MTQxIDguODE0NDUgMC43NDQxNDEgOC4zNDU3SDIuMzc4OTFDMi4zNzg5MSA4LjYyNjk1IDIuNDE5OTIgOC44NjMyOCAyLjUwMTk1IDkuMDU0NjlDMi41ODM5OCA5LjI0MjE5IDIuNjg5NDUgOS4zOTI1OCAyLjgxODM2IDkuNTA1ODZDMi45NTExNyA5LjYxNTIzIDMuMTAxNTYgOS42OTMzNiAzLjI2OTUzIDkuNzQwMjNDMy40Mzc1IDkuNzg3MTEgMy42MDkzOCA5LjgxMDU1IDMuNzg1MTYgOS44MTA1NUM0LjIwMzEyIDkuODEwNTUgNC41MTk1MyA5LjcxMjg5IDQuNzM0MzggOS41MTc1OEM0Ljk0OTIyIDkuMzIyMjcgNS4wNTY2NCA5LjA3MDMxIDUuMDU2NjQgOC43NjE3MlpNMTMuNDE4IDEyLjI3MTVIOC4wNzQyMlYxMUgxMy40MThWMTIuMjcxNVoiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMuOTUyNjQgNikiIGZpbGw9IndoaXRlIi8+Cjwvc3ZnPgo=);
  --jp-icon-text-editor: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtdGV4dC1lZGl0b3ItaWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xNSAxNUgzdjJoMTJ2LTJ6bTAtOEgzdjJoMTJWN3pNMyAxM2gxOHYtMkgzdjJ6bTAgOGgxOHYtMkgzdjJ6TTMgM3YyaDE4VjNIM3oiLz4KPC9zdmc+Cg==);
  --jp-icon-toc: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik03LDVIMjFWN0g3VjVNNywxM1YxMUgyMVYxM0g3TTQsNC41QTEuNSwxLjUgMCAwLDEgNS41LDZBMS41LDEuNSAwIDAsMSA0LDcuNUExLjUsMS41IDAgMCwxIDIuNSw2QTEuNSwxLjUgMCAwLDEgNCw0LjVNNCwxMC41QTEuNSwxLjUgMCAwLDEgNS41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMy41QTEuNSwxLjUgMCAwLDEgMi41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMC41TTcsMTlWMTdIMjFWMTlIN000LDE2LjVBMS41LDEuNSAwIDAsMSA1LjUsMThBMS41LDEuNSAwIDAsMSA0LDE5LjVBMS41LDEuNSAwIDAsMSAyLjUsMThBMS41LDEuNSAwIDAsMSA0LDE2LjVaIiAvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tree-view: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMiAxMVYzaC03djNIOVYzSDJ2OGg3VjhoMnYxMGg0djNoN3YtOGgtN3YzaC0yVjhoMnYzeiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMiAxNy4xODQ0IDIuOTY5NjggMTQuMzAzMiAxLjg2MDk0IDExLjQ0MDlaIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiMzMzMzMzMiIHN0cm9rZT0iIzMzMzMzMyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOCA5Ljg2NzE5KSIgZD0iTTIuODYwMTUgNC44NjUzNUwwLjcyNjU0OSAyLjk5OTU5TDAgMy42MzA0NUwyLjg2MDE1IDYuMTMxNTdMOCAwLjYzMDg3Mkw3LjI3ODU3IDBMMi44NjAxNSA0Ljg2NTM1WiIvPgo8L3N2Zz4K);
  --jp-icon-undo: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjUgOGMtMi42NSAwLTUuMDUuOTktNi45IDIuNkwyIDd2OWg5bC0zLjYyLTMuNjJjMS4zOS0xLjE2IDMuMTYtMS44OCA1LjEyLTEuODggMy41NCAwIDYuNTUgMi4zMSA3LjYgNS41bDIuMzctLjc4QzIxLjA4IDExLjAzIDE3LjE1IDggMTIuNSA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-user: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE2IDdhNCA0IDAgMTEtOCAwIDQgNCAwIDAxOCAwek0xMiAxNGE3IDcgMCAwMC03IDdoMTRhNyA3IDAgMDAtNy03eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-users: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZlcnNpb249IjEuMSIgdmlld0JveD0iMCAwIDM2IDI0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogPGcgY2xhc3M9ImpwLWljb24zIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjczMjcgMCAwIDEuNzMyNyAtMy42MjgyIC4wOTk1NzcpIiBmaWxsPSIjNjE2MTYxIj4KICA8cGF0aCB0cmFuc2Zvcm09Im1hdHJpeCgxLjUsMCwwLDEuNSwwLC02KSIgZD0ibTEyLjE4NiA3LjUwOThjLTEuMDUzNSAwLTEuOTc1NyAwLjU2NjUtMi40Nzg1IDEuNDEwMiAwLjc1MDYxIDAuMzEyNzcgMS4zOTc0IDAuODI2NDggMS44NzMgMS40NzI3aDMuNDg2M2MwLTEuNTkyLTEuMjg4OS0yLjg4MjgtMi44ODA5LTIuODgyOHoiLz4KICA8cGF0aCBkPSJtMjAuNDY1IDIuMzg5NWEyLjE4ODUgMi4xODg1IDAgMCAxLTIuMTg4NCAyLjE4ODUgMi4xODg1IDIuMTg4NSAwIDAgMS0yLjE4ODUtMi4xODg1IDIuMTg4NSAyLjE4ODUgMCAwIDEgMi4xODg1LTIuMTg4NSAyLjE4ODUgMi4xODg1IDAgMCAxIDIuMTg4NCAyLjE4ODV6Ii8+CiAgPHBhdGggdHJhbnNmb3JtPSJtYXRyaXgoMS41LDAsMCwxLjUsMCwtNikiIGQ9Im0zLjU4OTggOC40MjE5Yy0xLjExMjYgMC0yLjAxMzcgMC45MDExMS0yLjAxMzcgMi4wMTM3aDIuODE0NWMwLjI2Nzk3LTAuMzczMDkgMC41OTA3LTAuNzA0MzUgMC45NTg5OC0wLjk3ODUyLTAuMzQ0MzMtMC42MTY4OC0xLjAwMzEtMS4wMzUyLTEuNzU5OC0xLjAzNTJ6Ii8+CiAgPHBhdGggZD0ibTYuOTE1NCA0LjYyM2ExLjUyOTQgMS41Mjk0IDAgMCAxLTEuNTI5NCAxLjUyOTQgMS41Mjk0IDEuNTI5NCAwIDAgMS0xLjUyOTQtMS41Mjk0IDEuNTI5NCAxLjUyOTQgMCAwIDEgMS41Mjk0LTEuNTI5NCAxLjUyOTQgMS41Mjk0IDAgMCAxIDEuNTI5NCAxLjUyOTR6Ii8+CiAgPHBhdGggZD0ibTYuMTM1IDEzLjUzNWMwLTMuMjM5MiAyLjYyNTktNS44NjUgNS44NjUtNS44NjUgMy4yMzkyIDAgNS44NjUgMi42MjU5IDUuODY1IDUuODY1eiIvPgogIDxjaXJjbGUgY3g9IjEyIiBjeT0iMy43Njg1IiByPSIyLjk2ODUiLz4KIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-vega: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbjEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjEyMTIxIj4KICAgIDxwYXRoIGQ9Ik0xMC42IDUuNGwyLjItMy4ySDIuMnY3LjNsNC02LjZ6Ii8+CiAgICA8cGF0aCBkPSJNMTUuOCAyLjJsLTQuNCA2LjZMNyA2LjNsLTQuOCA4djUuNWgxNy42VjIuMmgtNHptLTcgMTUuNEg1LjV2LTQuNGgzLjN2NC40em00LjQgMEg5LjhWOS44aDMuNHY3Ljh6bTQuNCAwaC0zLjRWNi41aDMuNHYxMS4xeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-word: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KIDxnIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzQxNDE0MSI+CiAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiA8L2c+CiA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSguNDMgLjA0MDEpIiBmaWxsPSIjZmZmIj4KICA8cGF0aCBkPSJtNC4xNCA4Ljc2cTAuMDY4Mi0xLjg5IDIuNDItMS44OSAxLjE2IDAgMS42OCAwLjQyIDAuNTY3IDAuNDEgMC41NjcgMS4xNnYzLjQ3cTAgMC40NjIgMC41MTQgMC40NjIgMC4xMDMgMCAwLjItMC4wMjMxdjAuNzE0cS0wLjM5OSAwLjEwMy0wLjY1MSAwLjEwMy0wLjQ1MiAwLTAuNjkzLTAuMjItMC4yMzEtMC4yLTAuMjg0LTAuNjYyLTAuOTU2IDAuODcyLTIgMC44NzItMC45MDMgMC0xLjQ3LTAuNDcyLTAuNTI1LTAuNDcyLTAuNTI1LTEuMjYgMC0wLjI2MiAwLjA0NTItMC40NzIgMC4wNTY3LTAuMjIgMC4xMTYtMC4zNzggMC4wNjgyLTAuMTY4IDAuMjMxLTAuMzA0IDAuMTU4LTAuMTQ3IDAuMjYyLTAuMjQyIDAuMTE2LTAuMDkxNCAwLjM2OC0wLjE2OCAwLjI2Mi0wLjA5MTQgMC4zOTktMC4xMjYgMC4xMzYtMC4wNDUyIDAuNDcyLTAuMTAzIDAuMzM2LTAuMDU3OCAwLjUwNC0wLjA3OTggMC4xNTgtMC4wMjMxIDAuNTY3LTAuMDc5OCAwLjU1Ni0wLjA2ODIgMC43NzctMC4yMjEgMC4yMi0wLjE1MiAwLjIyLTAuNDQxdi0wLjI1MnEwLTAuNDMtMC4zNTctMC42NjItMC4zMzYtMC4yMzEtMC45NzYtMC4yMzEtMC42NjIgMC0wLjk5OCAwLjI2Mi0wLjMzNiAwLjI1Mi0wLjM5OSAwLjc5OHptMS44OSAzLjY4cTAuNzg4IDAgMS4yNi0wLjQxIDAuNTA0LTAuNDIgMC41MDQtMC45MDN2LTEuMDVxLTAuMjg0IDAuMTM2LTAuODYxIDAuMjMxLTAuNTY3IDAuMDkxNC0wLjk4NyAwLjE1OC0wLjQyIDAuMDY4Mi0wLjc2NiAwLjMyNi0wLjMzNiAwLjI1Mi0wLjMzNiAwLjcwNHQwLjMwNCAwLjcwNCAwLjg2MSAwLjI1MnoiIHN0cm9rZS13aWR0aD0iMS4wNSIvPgogIDxwYXRoIGQ9Im0xMCA0LjU2aDAuOTQ1djMuMTVxMC42NTEtMC45NzYgMS44OS0wLjk3NiAxLjE2IDAgMS44OSAwLjg0IDAuNjgyIDAuODQgMC42ODIgMi4zMSAwIDEuNDctMC43MDQgMi40Mi0wLjcwNCAwLjg4Mi0xLjg5IDAuODgyLTEuMjYgMC0xLjg5LTEuMDJ2MC43NjZoLTAuODV6bTIuNjIgMy4wNHEtMC43NDYgMC0xLjE2IDAuNjQtMC40NTIgMC42My0wLjQ1MiAxLjY4IDAgMS4wNSAwLjQ1MiAxLjY4dDEuMTYgMC42M3EwLjc3NyAwIDEuMjYtMC42MyAwLjQ5NC0wLjY0IDAuNDk0LTEuNjggMC0xLjA1LTAuNDcyLTEuNjgtMC40NjItMC42NC0xLjI2LTAuNjR6IiBzdHJva2Utd2lkdGg9IjEuMDUiLz4KICA8cGF0aCBkPSJtMi43MyAxNS44IDEzLjYgMC4wMDgxYzAuMDA2OSAwIDAtMi42IDAtMi42IDAtMC4wMDc4LTEuMTUgMC0xLjE1IDAtMC4wMDY5IDAtMC4wMDgzIDEuNS0wLjAwODMgMS41LTJlLTMgLTAuMDAxNC0xMS4zLTAuMDAxNC0xMS4zLTAuMDAxNGwtMC4wMDU5Mi0xLjVjMC0wLjAwNzgtMS4xNyAwLjAwMTMtMS4xNyAwLjAwMTN6IiBzdHJva2Utd2lkdGg9Ii45NzUiLz4KIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-yaml: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1jb250cmFzdDIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjRDgxQjYwIj4KICAgIDxwYXRoIGQ9Ik03LjIgMTguNnYtNS40TDMgNS42aDMuM2wxLjQgMy4xYy4zLjkuNiAxLjYgMSAyLjUuMy0uOC42LTEuNiAxLTIuNWwxLjQtMy4xaDMuNGwtNC40IDcuNnY1LjVsLTIuOS0uMXoiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxNi41IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxMSIgcj0iMi4xIi8+CiAgPC9nPgo8L3N2Zz4K);
}

/* Icon CSS class declarations */

.jp-AddAboveIcon {
  background-image: var(--jp-icon-add-above);
}

.jp-AddBelowIcon {
  background-image: var(--jp-icon-add-below);
}

.jp-AddIcon {
  background-image: var(--jp-icon-add);
}

.jp-BellIcon {
  background-image: var(--jp-icon-bell);
}

.jp-BugDotIcon {
  background-image: var(--jp-icon-bug-dot);
}

.jp-BugIcon {
  background-image: var(--jp-icon-bug);
}

.jp-BuildIcon {
  background-image: var(--jp-icon-build);
}

.jp-CaretDownEmptyIcon {
  background-image: var(--jp-icon-caret-down-empty);
}

.jp-CaretDownEmptyThinIcon {
  background-image: var(--jp-icon-caret-down-empty-thin);
}

.jp-CaretDownIcon {
  background-image: var(--jp-icon-caret-down);
}

.jp-CaretLeftIcon {
  background-image: var(--jp-icon-caret-left);
}

.jp-CaretRightIcon {
  background-image: var(--jp-icon-caret-right);
}

.jp-CaretUpEmptyThinIcon {
  background-image: var(--jp-icon-caret-up-empty-thin);
}

.jp-CaretUpIcon {
  background-image: var(--jp-icon-caret-up);
}

.jp-CaseSensitiveIcon {
  background-image: var(--jp-icon-case-sensitive);
}

.jp-CheckIcon {
  background-image: var(--jp-icon-check);
}

.jp-CircleEmptyIcon {
  background-image: var(--jp-icon-circle-empty);
}

.jp-CircleIcon {
  background-image: var(--jp-icon-circle);
}

.jp-ClearIcon {
  background-image: var(--jp-icon-clear);
}

.jp-CloseIcon {
  background-image: var(--jp-icon-close);
}

.jp-CodeCheckIcon {
  background-image: var(--jp-icon-code-check);
}

.jp-CodeIcon {
  background-image: var(--jp-icon-code);
}

.jp-CollapseAllIcon {
  background-image: var(--jp-icon-collapse-all);
}

.jp-ConsoleIcon {
  background-image: var(--jp-icon-console);
}

.jp-CopyIcon {
  background-image: var(--jp-icon-copy);
}

.jp-CopyrightIcon {
  background-image: var(--jp-icon-copyright);
}

.jp-CutIcon {
  background-image: var(--jp-icon-cut);
}

.jp-DeleteIcon {
  background-image: var(--jp-icon-delete);
}

.jp-DownloadIcon {
  background-image: var(--jp-icon-download);
}

.jp-DuplicateIcon {
  background-image: var(--jp-icon-duplicate);
}

.jp-EditIcon {
  background-image: var(--jp-icon-edit);
}

.jp-EllipsesIcon {
  background-image: var(--jp-icon-ellipses);
}

.jp-ErrorIcon {
  background-image: var(--jp-icon-error);
}

.jp-ExpandAllIcon {
  background-image: var(--jp-icon-expand-all);
}

.jp-ExtensionIcon {
  background-image: var(--jp-icon-extension);
}

.jp-FastForwardIcon {
  background-image: var(--jp-icon-fast-forward);
}

.jp-FileIcon {
  background-image: var(--jp-icon-file);
}

.jp-FileUploadIcon {
  background-image: var(--jp-icon-file-upload);
}

.jp-FilterDotIcon {
  background-image: var(--jp-icon-filter-dot);
}

.jp-FilterIcon {
  background-image: var(--jp-icon-filter);
}

.jp-FilterListIcon {
  background-image: var(--jp-icon-filter-list);
}

.jp-FolderFavoriteIcon {
  background-image: var(--jp-icon-folder-favorite);
}

.jp-FolderIcon {
  background-image: var(--jp-icon-folder);
}

.jp-HomeIcon {
  background-image: var(--jp-icon-home);
}

.jp-Html5Icon {
  background-image: var(--jp-icon-html5);
}

.jp-ImageIcon {
  background-image: var(--jp-icon-image);
}

.jp-InfoIcon {
  background-image: var(--jp-icon-info);
}

.jp-InspectorIcon {
  background-image: var(--jp-icon-inspector);
}

.jp-JsonIcon {
  background-image: var(--jp-icon-json);
}

.jp-JuliaIcon {
  background-image: var(--jp-icon-julia);
}

.jp-JupyterFaviconIcon {
  background-image: var(--jp-icon-jupyter-favicon);
}

.jp-JupyterIcon {
  background-image: var(--jp-icon-jupyter);
}

.jp-JupyterlabWordmarkIcon {
  background-image: var(--jp-icon-jupyterlab-wordmark);
}

.jp-KernelIcon {
  background-image: var(--jp-icon-kernel);
}

.jp-KeyboardIcon {
  background-image: var(--jp-icon-keyboard);
}

.jp-LaunchIcon {
  background-image: var(--jp-icon-launch);
}

.jp-LauncherIcon {
  background-image: var(--jp-icon-launcher);
}

.jp-LineFormIcon {
  background-image: var(--jp-icon-line-form);
}

.jp-LinkIcon {
  background-image: var(--jp-icon-link);
}

.jp-ListIcon {
  background-image: var(--jp-icon-list);
}

.jp-MarkdownIcon {
  background-image: var(--jp-icon-markdown);
}

.jp-MoveDownIcon {
  background-image: var(--jp-icon-move-down);
}

.jp-MoveUpIcon {
  background-image: var(--jp-icon-move-up);
}

.jp-NewFolderIcon {
  background-image: var(--jp-icon-new-folder);
}

.jp-NotTrustedIcon {
  background-image: var(--jp-icon-not-trusted);
}

.jp-NotebookIcon {
  background-image: var(--jp-icon-notebook);
}

.jp-NumberingIcon {
  background-image: var(--jp-icon-numbering);
}

.jp-OfflineBoltIcon {
  background-image: var(--jp-icon-offline-bolt);
}

.jp-PaletteIcon {
  background-image: var(--jp-icon-palette);
}

.jp-PasteIcon {
  background-image: var(--jp-icon-paste);
}

.jp-PdfIcon {
  background-image: var(--jp-icon-pdf);
}

.jp-PythonIcon {
  background-image: var(--jp-icon-python);
}

.jp-RKernelIcon {
  background-image: var(--jp-icon-r-kernel);
}

.jp-ReactIcon {
  background-image: var(--jp-icon-react);
}

.jp-RedoIcon {
  background-image: var(--jp-icon-redo);
}

.jp-RefreshIcon {
  background-image: var(--jp-icon-refresh);
}

.jp-RegexIcon {
  background-image: var(--jp-icon-regex);
}

.jp-RunIcon {
  background-image: var(--jp-icon-run);
}

.jp-RunningIcon {
  background-image: var(--jp-icon-running);
}

.jp-SaveIcon {
  background-image: var(--jp-icon-save);
}

.jp-SearchIcon {
  background-image: var(--jp-icon-search);
}

.jp-SettingsIcon {
  background-image: var(--jp-icon-settings);
}

.jp-ShareIcon {
  background-image: var(--jp-icon-share);
}

.jp-SpreadsheetIcon {
  background-image: var(--jp-icon-spreadsheet);
}

.jp-StopIcon {
  background-image: var(--jp-icon-stop);
}

.jp-TabIcon {
  background-image: var(--jp-icon-tab);
}

.jp-TableRowsIcon {
  background-image: var(--jp-icon-table-rows);
}

.jp-TagIcon {
  background-image: var(--jp-icon-tag);
}

.jp-TerminalIcon {
  background-image: var(--jp-icon-terminal);
}

.jp-TextEditorIcon {
  background-image: var(--jp-icon-text-editor);
}

.jp-TocIcon {
  background-image: var(--jp-icon-toc);
}

.jp-TreeViewIcon {
  background-image: var(--jp-icon-tree-view);
}

.jp-TrustedIcon {
  background-image: var(--jp-icon-trusted);
}

.jp-UndoIcon {
  background-image: var(--jp-icon-undo);
}

.jp-UserIcon {
  background-image: var(--jp-icon-user);
}

.jp-UsersIcon {
  background-image: var(--jp-icon-users);
}

.jp-VegaIcon {
  background-image: var(--jp-icon-vega);
}

.jp-WordIcon {
  background-image: var(--jp-icon-word);
}

.jp-YamlIcon {
  background-image: var(--jp-icon-yaml);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

.jp-Icon,
.jp-MaterialIcon {
  background-position: center;
  background-repeat: no-repeat;
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-cover {
  background-position: center;
  background-repeat: no-repeat;
  background-size: cover;
}

/**
 * (DEPRECATED) Support for specific CSS icon sizes
 */

.jp-Icon-16 {
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-18 {
  background-size: 18px;
  min-width: 18px;
  min-height: 18px;
}

.jp-Icon-20 {
  background-size: 20px;
  min-width: 20px;
  min-height: 20px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.lm-TabBar .lm-TabBar-addButton {
  align-items: center;
  display: flex;
  padding: 4px;
  padding-bottom: 5px;
  margin-right: 1px;
  background-color: var(--jp-layout-color2);
}

.lm-TabBar .lm-TabBar-addButton:hover {
  background-color: var(--jp-layout-color1);
}

.lm-DockPanel-tabBar .lm-TabBar-tab {
  width: var(--jp-private-horizontal-tab-width);
}

.lm-DockPanel-tabBar .lm-TabBar-content {
  flex: unset;
}

.lm-DockPanel-tabBar[data-orientation='horizontal'] {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for icons as inline SVG HTMLElements
 */

/* recolor the primary elements of an icon */
.jp-icon0[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon1[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon2[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon3[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-accent0[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-accent1[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-accent2[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-accent3[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-accent4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-accent0[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-accent1[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-accent2[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-accent3[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-accent4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-none[fill] {
  fill: none;
}

.jp-icon-none[stroke] {
  stroke: none;
}

/* brand icon colors. Same for light and dark */
.jp-icon-brand0[fill] {
  fill: var(--jp-brand-color0);
}

.jp-icon-brand1[fill] {
  fill: var(--jp-brand-color1);
}

.jp-icon-brand2[fill] {
  fill: var(--jp-brand-color2);
}

.jp-icon-brand3[fill] {
  fill: var(--jp-brand-color3);
}

.jp-icon-brand4[fill] {
  fill: var(--jp-brand-color4);
}

.jp-icon-brand0[stroke] {
  stroke: var(--jp-brand-color0);
}

.jp-icon-brand1[stroke] {
  stroke: var(--jp-brand-color1);
}

.jp-icon-brand2[stroke] {
  stroke: var(--jp-brand-color2);
}

.jp-icon-brand3[stroke] {
  stroke: var(--jp-brand-color3);
}

.jp-icon-brand4[stroke] {
  stroke: var(--jp-brand-color4);
}

/* warn icon colors. Same for light and dark */
.jp-icon-warn0[fill] {
  fill: var(--jp-warn-color0);
}

.jp-icon-warn1[fill] {
  fill: var(--jp-warn-color1);
}

.jp-icon-warn2[fill] {
  fill: var(--jp-warn-color2);
}

.jp-icon-warn3[fill] {
  fill: var(--jp-warn-color3);
}

.jp-icon-warn0[stroke] {
  stroke: var(--jp-warn-color0);
}

.jp-icon-warn1[stroke] {
  stroke: var(--jp-warn-color1);
}

.jp-icon-warn2[stroke] {
  stroke: var(--jp-warn-color2);
}

.jp-icon-warn3[stroke] {
  stroke: var(--jp-warn-color3);
}

/* icon colors that contrast well with each other and most backgrounds */
.jp-icon-contrast0[fill] {
  fill: var(--jp-icon-contrast-color0);
}

.jp-icon-contrast1[fill] {
  fill: var(--jp-icon-contrast-color1);
}

.jp-icon-contrast2[fill] {
  fill: var(--jp-icon-contrast-color2);
}

.jp-icon-contrast3[fill] {
  fill: var(--jp-icon-contrast-color3);
}

.jp-icon-contrast0[stroke] {
  stroke: var(--jp-icon-contrast-color0);
}

.jp-icon-contrast1[stroke] {
  stroke: var(--jp-icon-contrast-color1);
}

.jp-icon-contrast2[stroke] {
  stroke: var(--jp-icon-contrast-color2);
}

.jp-icon-contrast3[stroke] {
  stroke: var(--jp-icon-contrast-color3);
}

.jp-icon-dot[fill] {
  fill: var(--jp-warn-color0);
}

.jp-jupyter-icon-color[fill] {
  fill: var(--jp-jupyter-icon-color, var(--jp-warn-color0));
}

.jp-notebook-icon-color[fill] {
  fill: var(--jp-notebook-icon-color, var(--jp-warn-color0));
}

.jp-json-icon-color[fill] {
  fill: var(--jp-json-icon-color, var(--jp-warn-color1));
}

.jp-console-icon-color[fill] {
  fill: var(--jp-console-icon-color, white);
}

.jp-console-icon-background-color[fill] {
  fill: var(--jp-console-icon-background-color, var(--jp-brand-color1));
}

.jp-terminal-icon-color[fill] {
  fill: var(--jp-terminal-icon-color, var(--jp-layout-color2));
}

.jp-terminal-icon-background-color[fill] {
  fill: var(
    --jp-terminal-icon-background-color,
    var(--jp-inverse-layout-color2)
  );
}

.jp-text-editor-icon-color[fill] {
  fill: var(--jp-text-editor-icon-color, var(--jp-inverse-layout-color3));
}

.jp-inspector-icon-color[fill] {
  fill: var(--jp-inspector-icon-color, var(--jp-inverse-layout-color3));
}

/* CSS for icons in selected filebrowser listing items */
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

.jp-DirListing-item.jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* stylelint-disable selector-max-class, selector-max-compound-selectors */

/**
* TODO: come up with non css-hack solution for showing the busy icon on top
*  of the close icon
* CSS for complex behavior of close icon of tabs in the main area tabbar
*/
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}

.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

/* stylelint-enable selector-max-class, selector-max-compound-selectors */

/* CSS for icons in status bar */
#jp-main-statusbar .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

#jp-main-statusbar .jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* special handling for splash icon CSS. While the theme CSS reloads during
   splash, the splash icon can loose theming. To prevent that, we set a
   default for its color variable */
:root {
  --jp-warn-color0: var(--md-orange-700);
}

/* not sure what to do with this one, used in filebrowser listing */
.jp-DragIcon {
  margin-right: 4px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for alt colors for icons as inline SVG HTMLElements
 */

/* alt recolor the primary elements of an icon */
.jp-icon-alt .jp-icon0[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-alt .jp-icon1[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-alt .jp-icon2[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-alt .jp-icon3[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-alt .jp-icon4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-alt .jp-icon0[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-alt .jp-icon1[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-alt .jp-icon2[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-alt .jp-icon3[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-alt .jp-icon4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* alt recolor the accent elements of an icon */
.jp-icon-alt .jp-icon-accent0[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-alt .jp-icon-accent1[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-alt .jp-icon-accent2[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-alt .jp-icon-accent3[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-alt .jp-icon-accent4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-alt .jp-icon-accent0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-alt .jp-icon-accent1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-alt .jp-icon-accent2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-alt .jp-icon-accent3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-alt .jp-icon-accent4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-icon-hoverShow:not(:hover) .jp-icon-hoverShow-content {
  display: none !important;
}

/**
 * Support for hover colors for icons as inline SVG HTMLElements
 */

/**
 * regular colors
 */

/* recolor the primary elements of an icon */
.jp-icon-hover :hover .jp-icon0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-hover :hover .jp-icon1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-hover :hover .jp-icon2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-hover :hover .jp-icon3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-hover :hover .jp-icon4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-hover :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-hover :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-hover :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-hover :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-hover :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-hover :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-hover :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-hover :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-hover :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-hover :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-hover :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-hover :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-hover :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-hover :hover .jp-icon-none-hover[fill] {
  fill: none;
}

.jp-icon-hover :hover .jp-icon-none-hover[stroke] {
  stroke: none;
}

/**
 * inverse colors
 */

/* inverse recolor the primary elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* inverse recolor the accent elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-IFrame {
  width: 100%;
  height: 100%;
}

.jp-IFrame > iframe {
  border: none;
}

/*
When drag events occur, `lm-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-IFrame {
  position: relative;
}

body.lm-mod-override-cursor .jp-IFrame::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-HoverBox {
  position: fixed;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FormGroup-content fieldset {
  border: none;
  padding: 0;
  min-width: 0;
  width: 100%;
}

/* stylelint-disable selector-max-type */

.jp-FormGroup-content fieldset .jp-inputFieldWrapper input,
.jp-FormGroup-content fieldset .jp-inputFieldWrapper select,
.jp-FormGroup-content fieldset .jp-inputFieldWrapper textarea {
  font-size: var(--jp-content-font-size2);
  border-color: var(--jp-input-border-color);
  border-style: solid;
  border-radius: var(--jp-border-radius);
  border-width: 1px;
  padding: 6px 8px;
  background: none;
  color: var(--jp-ui-font-color0);
  height: inherit;
}

.jp-FormGroup-content fieldset input[type='checkbox'] {
  position: relative;
  top: 2px;
  margin-left: 0;
}

.jp-FormGroup-content button.jp-mod-styled {
  cursor: pointer;
}

.jp-FormGroup-content .checkbox label {
  cursor: pointer;
  font-size: var(--jp-content-font-size1);
}

.jp-FormGroup-content .jp-root > fieldset > legend {
  display: none;
}

.jp-FormGroup-content .jp-root > fieldset > p {
  display: none;
}

/** copy of `input.jp-mod-styled:focus` style */
.jp-FormGroup-content fieldset input:focus,
.jp-FormGroup-content fieldset select:focus {
  -moz-outline-radius: unset;
  outline: var(--jp-border-width) solid var(--md-blue-500);
  outline-offset: -1px;
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-FormGroup-content fieldset input:hover:not(:focus),
.jp-FormGroup-content fieldset select:hover:not(:focus) {
  background-color: var(--jp-border-color2);
}

/* stylelint-enable selector-max-type */

.jp-FormGroup-content .checkbox .field-description {
  /* Disable default description field for checkbox:
   because other widgets do not have description fields,
   we add descriptions to each widget on the field level.
  */
  display: none;
}

.jp-FormGroup-content #root__description {
  display: none;
}

.jp-FormGroup-content .jp-modifiedIndicator {
  width: 5px;
  background-color: var(--jp-brand-color2);
  margin-top: 0;
  margin-left: calc(var(--jp-private-settingeditor-modifier-indent) * -1);
  flex-shrink: 0;
}

.jp-FormGroup-content .jp-modifiedIndicator.jp-errorIndicator {
  background-color: var(--jp-error-color0);
  margin-right: 0.5em;
}

/* RJSF ARRAY style */

.jp-arrayFieldWrapper legend {
  font-size: var(--jp-content-font-size2);
  color: var(--jp-ui-font-color0);
  flex-basis: 100%;
  padding: 4px 0;
  font-weight: var(--jp-content-heading-font-weight);
  border-bottom: 1px solid var(--jp-border-color2);
}

.jp-arrayFieldWrapper .field-description {
  padding: 4px 0;
  white-space: pre-wrap;
}

.jp-arrayFieldWrapper .array-item {
  width: 100%;
  border: 1px solid var(--jp-border-color2);
  border-radius: 4px;
  margin: 4px;
}

.jp-ArrayOperations {
  display: flex;
  margin-left: 8px;
}

.jp-ArrayOperationsButton {
  margin: 2px;
}

.jp-ArrayOperationsButton .jp-icon3[fill] {
  fill: var(--jp-ui-font-color0);
}

button.jp-ArrayOperationsButton.jp-mod-styled:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

/* RJSF form validation error */

.jp-FormGroup-content .validationErrors {
  color: var(--jp-error-color0);
}

/* Hide panel level error as duplicated the field level error */
.jp-FormGroup-content .panel.errors {
  display: none;
}

/* RJSF normal content (settings-editor) */

.jp-FormGroup-contentNormal {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.jp-FormGroup-contentNormal .jp-FormGroup-contentItem {
  margin-left: 7px;
  color: var(--jp-ui-font-color0);
}

.jp-FormGroup-contentNormal .jp-FormGroup-description {
  flex-basis: 100%;
  padding: 4px 7px;
}

.jp-FormGroup-contentNormal .jp-FormGroup-default {
  flex-basis: 100%;
  padding: 4px 7px;
}

.jp-FormGroup-contentNormal .jp-FormGroup-fieldLabel {
  font-size: var(--jp-content-font-size1);
  font-weight: normal;
  min-width: 120px;
}

.jp-FormGroup-contentNormal fieldset:not(:first-child) {
  margin-left: 7px;
}

.jp-FormGroup-contentNormal .field-array-of-string .array-item {
  /* Display `jp-ArrayOperations` buttons side-by-side with content except
    for small screens where flex-wrap will place them one below the other.
  */
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.jp-FormGroup-contentNormal .jp-objectFieldWrapper .form-group {
  padding: 2px 8px 2px var(--jp-private-settingeditor-modifier-indent);
  margin-top: 2px;
}

/* RJSF compact content (metadata-form) */

.jp-FormGroup-content.jp-FormGroup-contentCompact {
  width: 100%;
}

.jp-FormGroup-contentCompact .form-group {
  display: flex;
  padding: 0.5em 0.2em 0.5em 0;
}

.jp-FormGroup-contentCompact
  .jp-FormGroup-compactTitle
  .jp-FormGroup-description {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color2);
}

.jp-FormGroup-contentCompact .jp-FormGroup-fieldLabel {
  padding-bottom: 0.3em;
}

.jp-FormGroup-contentCompact .jp-inputFieldWrapper .form-control {
  width: 100%;
  box-sizing: border-box;
}

.jp-FormGroup-contentCompact .jp-arrayFieldWrapper .jp-FormGroup-compactTitle {
  padding-bottom: 7px;
}

.jp-FormGroup-contentCompact
  .jp-objectFieldWrapper
  .jp-objectFieldWrapper
  .form-group {
  padding: 2px 8px 2px var(--jp-private-settingeditor-modifier-indent);
  margin-top: 2px;
}

.jp-FormGroup-contentCompact ul.error-detail {
  margin-block-start: 0.5em;
  margin-block-end: 0.5em;
  padding-inline-start: 1em;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-SidePanel {
  display: flex;
  flex-direction: column;
  min-width: var(--jp-sidebar-min-width);
  overflow-y: auto;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  font-size: var(--jp-ui-font-size1);
}

.jp-SidePanel-header {
  flex: 0 0 auto;
  display: flex;
  border-bottom: var(--jp-border-width) solid var(--jp-border-color2);
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin: 0;
  padding: 2px;
  text-transform: uppercase;
}

.jp-SidePanel-toolbar {
  flex: 0 0 auto;
}

.jp-SidePanel-content {
  flex: 1 1 auto;
}

.jp-SidePanel-toolbar,
.jp-AccordionPanel-toolbar {
  height: var(--jp-private-toolbar-height);
}

.jp-SidePanel-toolbar.jp-Toolbar-micro {
  display: none;
}

.lm-AccordionPanel .jp-AccordionPanel-title {
  box-sizing: border-box;
  line-height: 25px;
  margin: 0;
  display: flex;
  align-items: center;
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  font-size: var(--jp-ui-font-size0);
}

.jp-AccordionPanel-title {
  cursor: pointer;
  user-select: none;
  -moz-user-select: none;
  -webkit-user-select: none;
  text-transform: uppercase;
}

.lm-AccordionPanel[data-orientation='horizontal'] > .jp-AccordionPanel-title {
  /* Title is rotated for horizontal accordion panel using CSS */
  display: block;
  transform-origin: top left;
  transform: rotate(-90deg) translate(-100%);
}

.jp-AccordionPanel-title .lm-AccordionPanel-titleLabel {
  user-select: none;
  text-overflow: ellipsis;
  white-space: nowrap;
  overflow: hidden;
}

.jp-AccordionPanel-title .lm-AccordionPanel-titleCollapser {
  transform: rotate(-90deg);
  margin: auto 0;
  height: 16px;
}

.jp-AccordionPanel-title.lm-mod-expanded .lm-AccordionPanel-titleCollapser {
  transform: rotate(0deg);
}

.lm-AccordionPanel .jp-AccordionPanel-toolbar {
  background: none;
  box-shadow: none;
  border: none;
  margin-left: auto;
}

.lm-AccordionPanel .lm-SplitPanel-handle:hover {
  background: var(--jp-layout-color3);
}

.jp-text-truncated {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Spinner {
  position: absolute;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 10;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-layout-color0);
  outline: none;
}

.jp-SpinnerContent {
  font-size: 10px;
  margin: 50px auto;
  text-indent: -9999em;
  width: 3em;
  height: 3em;
  border-radius: 50%;
  background: var(--jp-brand-color3);
  background: linear-gradient(
    to right,
    #f37626 10%,
    rgba(255, 255, 255, 0) 42%
  );
  position: relative;
  animation: load3 1s infinite linear, fadeIn 1s;
}

.jp-SpinnerContent::before {
  width: 50%;
  height: 50%;
  background: #f37626;
  border-radius: 100% 0 0;
  position: absolute;
  top: 0;
  left: 0;
  content: '';
}

.jp-SpinnerContent::after {
  background: var(--jp-layout-color0);
  width: 75%;
  height: 75%;
  border-radius: 50%;
  content: '';
  margin: auto;
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
}

@keyframes fadeIn {
  0% {
    opacity: 0;
  }

  100% {
    opacity: 1;
  }
}

@keyframes load3 {
  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

button.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: none;
  box-sizing: border-box;
  text-align: center;
  line-height: 32px;
  height: 32px;
  padding: 0 12px;
  letter-spacing: 0.8px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input.jp-mod-styled {
  background: var(--jp-input-background);
  height: 28px;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color1);
  padding-left: 7px;
  padding-right: 7px;
  font-size: var(--jp-ui-font-size2);
  color: var(--jp-ui-font-color0);
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input[type='checkbox'].jp-mod-styled {
  appearance: checkbox;
  -webkit-appearance: checkbox;
  -moz-appearance: checkbox;
  height: auto;
}

input.jp-mod-styled:focus {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-select-wrapper {
  display: flex;
  position: relative;
  flex-direction: column;
  padding: 1px;
  background-color: var(--jp-layout-color1);
  box-sizing: border-box;
  margin-bottom: 12px;
}

.jp-select-wrapper:not(.multiple) {
  height: 28px;
}

.jp-select-wrapper.jp-mod-focused select.jp-mod-styled {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-input-active-background);
}

select.jp-mod-styled:hover {
  cursor: pointer;
  color: var(--jp-ui-font-color0);
  background-color: var(--jp-input-hover-background);
  box-shadow: inset 0 0 1px rgba(0, 0, 0, 0.5);
}

select.jp-mod-styled {
  flex: 1 1 auto;
  width: 100%;
  font-size: var(--jp-ui-font-size2);
  background: var(--jp-input-background);
  color: var(--jp-ui-font-color0);
  padding: 0 25px 0 8px;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

select.jp-mod-styled:not([multiple]) {
  height: 32px;
}

select.jp-mod-styled[multiple] {
  max-height: 200px;
  overflow-y: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-switch {
  display: flex;
  align-items: center;
  padding-left: 4px;
  padding-right: 4px;
  font-size: var(--jp-ui-font-size1);
  background-color: transparent;
  color: var(--jp-ui-font-color1);
  border: none;
  height: 20px;
}

.jp-switch:hover {
  background-color: var(--jp-layout-color2);
}

.jp-switch-label {
  margin-right: 5px;
  font-family: var(--jp-ui-font-family);
}

.jp-switch-track {
  cursor: pointer;
  background-color: var(--jp-switch-color, var(--jp-border-color1));
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 34px;
  height: 16px;
  width: 35px;
  position: relative;
}

.jp-switch-track::before {
  content: '';
  position: absolute;
  height: 10px;
  width: 10px;
  margin: 3px;
  left: 0;
  background-color: var(--jp-ui-inverse-font-color1);
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 50%;
}

.jp-switch[aria-checked='true'] .jp-switch-track {
  background-color: var(--jp-switch-true-position-color, var(--jp-warn-color0));
}

.jp-switch[aria-checked='true'] .jp-switch-track::before {
  /* track width (35) - margins (3 + 3) - thumb width (10) */
  left: 19px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toolbar-height: calc(
    28px + var(--jp-border-width)
  ); /* leave 28px for content */
}

.jp-Toolbar {
  color: var(--jp-ui-font-color1);
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: 2px;
  z-index: 8;
  overflow-x: hidden;
}

/* Toolbar items */

.jp-Toolbar > .jp-Toolbar-item.jp-Toolbar-spacer {
  flex-grow: 1;
  flex-shrink: 1;
}

.jp-Toolbar-item.jp-Toolbar-kernelStatus {
  display: inline-block;
  width: 32px;
  background-repeat: no-repeat;
  background-position: center;
  background-size: 16px;
}

.jp-Toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  display: flex;
  padding-left: 1px;
  padding-right: 1px;
  font-size: var(--jp-ui-font-size1);
  line-height: var(--jp-private-toolbar-height);
  height: 100%;
}

/* Toolbar buttons */

/* This is the div we use to wrap the react component into a Widget */
div.jp-ToolbarButton {
  color: transparent;
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0;
  margin: 0;
}

button.jp-ToolbarButtonComponent {
  background: var(--jp-layout-color1);
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0 6px;
  margin: 0;
  height: 24px;
  border-radius: var(--jp-border-radius);
  display: flex;
  align-items: center;
  text-align: center;
  font-size: 14px;
  min-width: unset;
  min-height: unset;
}

button.jp-ToolbarButtonComponent:disabled {
  opacity: 0.4;
}

button.jp-ToolbarButtonComponent > span {
  padding: 0;
  flex: 0 0 auto;
}

button.jp-ToolbarButtonComponent .jp-ToolbarButtonComponent-label {
  font-size: var(--jp-ui-font-size1);
  line-height: 100%;
  padding-left: 2px;
  color: var(--jp-ui-font-color1);
  font-family: var(--jp-ui-font-family);
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar.jp-Toolbar-micro {
  padding: 0;
  min-height: 0;
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar {
  border: none;
  box-shadow: none;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-WindowedPanel-outer {
  position: relative;
  overflow-y: auto;
}

.jp-WindowedPanel-inner {
  position: relative;
}

.jp-WindowedPanel-window {
  position: absolute;
  left: 0;
  right: 0;
  overflow: visible;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* Sibling imports */

body {
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
}

/* Disable native link decoration styles everywhere outside of dialog boxes */
a {
  text-decoration: unset;
  color: unset;
}

a:hover {
  text-decoration: unset;
  color: unset;
}

/* Accessibility for links inside dialog box text */
.jp-Dialog-content a {
  text-decoration: revert;
  color: var(--jp-content-link-color);
}

.jp-Dialog-content a:hover {
  text-decoration: revert;
}

/* Styles for ui-components */
.jp-Button {
  color: var(--jp-ui-font-color2);
  border-radius: var(--jp-border-radius);
  padding: 0 12px;
  font-size: var(--jp-ui-font-size1);

  /* Copy from blueprint 3 */
  display: inline-flex;
  flex-direction: row;
  border: none;
  cursor: pointer;
  align-items: center;
  justify-content: center;
  text-align: left;
  vertical-align: middle;
  min-height: 30px;
  min-width: 30px;
}

.jp-Button:disabled {
  cursor: not-allowed;
}

.jp-Button:empty {
  padding: 0 !important;
}

.jp-Button.jp-mod-small {
  min-height: 24px;
  min-width: 24px;
  font-size: 12px;
  padding: 0 7px;
}

/* Use our own theme for hover styles */
.jp-Button.jp-mod-minimal:hover {
  background-color: var(--jp-layout-color2);
}

.jp-Button.jp-mod-minimal {
  background: none;
}

.jp-InputGroup {
  display: block;
  position: relative;
}

.jp-InputGroup input {
  box-sizing: border-box;
  border: none;
  border-radius: 0;
  background-color: transparent;
  color: var(--jp-ui-font-color0);
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
  padding-bottom: 0;
  padding-top: 0;
  padding-left: 10px;
  padding-right: 28px;
  position: relative;
  width: 100%;
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;
  font-size: 14px;
  font-weight: 400;
  height: 30px;
  line-height: 30px;
  outline: none;
  vertical-align: middle;
}

.jp-InputGroup input:focus {
  box-shadow: inset 0 0 0 var(--jp-border-width)
      var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-InputGroup input:disabled {
  cursor: not-allowed;
  resize: block;
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color2);
}

.jp-InputGroup input:disabled ~ span {
  cursor: not-allowed;
  color: var(--jp-ui-font-color2);
}

.jp-InputGroup input::placeholder,
input::placeholder {
  color: var(--jp-ui-font-color2);
}

.jp-InputGroupAction {
  position: absolute;
  bottom: 1px;
  right: 0;
  padding: 6px;
}

.jp-HTMLSelect.jp-DefaultStyle select {
  background-color: initial;
  border: none;
  border-radius: 0;
  box-shadow: none;
  color: var(--jp-ui-font-color0);
  display: block;
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  height: 24px;
  line-height: 14px;
  padding: 0 25px 0 10px;
  text-align: left;
  -moz-appearance: none;
  -webkit-appearance: none;
}

.jp-HTMLSelect.jp-DefaultStyle select:disabled {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color2);
  cursor: not-allowed;
  resize: block;
}

.jp-HTMLSelect.jp-DefaultStyle select:disabled ~ span {
  cursor: not-allowed;
}

/* Use our own theme for hover and option styles */
/* stylelint-disable-next-line selector-max-type */
.jp-HTMLSelect.jp-DefaultStyle select:hover,
.jp-HTMLSelect.jp-DefaultStyle select > option {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color0);
}

select {
  box-sizing: border-box;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-StatusBar-Widget {
  display: flex;
  align-items: center;
  background: var(--jp-layout-color2);
  min-height: var(--jp-statusbar-height);
  justify-content: space-between;
  padding: 0 10px;
}

.jp-StatusBar-Left {
  display: flex;
  align-items: center;
  flex-direction: row;
}

.jp-StatusBar-Middle {
  display: flex;
  align-items: center;
}

.jp-StatusBar-Right {
  display: flex;
  align-items: center;
  flex-direction: row-reverse;
}

.jp-StatusBar-Item {
  max-height: var(--jp-statusbar-height);
  margin: 0 2px;
  height: var(--jp-statusbar-height);
  white-space: nowrap;
  text-overflow: ellipsis;
  color: var(--jp-ui-font-color1);
  padding: 0 6px;
}

.jp-mod-highlighted:hover {
  background-color: var(--jp-layout-color3);
}

.jp-mod-clicked {
  background-color: var(--jp-brand-color1);
}

.jp-mod-clicked:hover {
  background-color: var(--jp-brand-color0);
}

.jp-mod-clicked .jp-StatusBar-TextItem {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-StatusBar-HoverItem {
  box-shadow: '0px 4px 4px rgba(0, 0, 0, 0.25)';
}

.jp-StatusBar-TextItem {
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  line-height: 24px;
  color: var(--jp-ui-font-color1);
}

.jp-StatusBar-GroupItem {
  display: flex;
  align-items: center;
  flex-direction: row;
}

.jp-Statusbar-ProgressCircle svg {
  display: block;
  margin: 0 auto;
  width: 16px;
  height: 24px;
  align-self: normal;
}

.jp-Statusbar-ProgressCircle path {
  fill: var(--jp-inverse-layout-color3);
}

.jp-Statusbar-ProgressBar-progress-bar {
  height: 10px;
  width: 100px;
  border: solid 0.25px var(--jp-brand-color2);
  border-radius: 3px;
  overflow: hidden;
  align-self: center;
}

.jp-Statusbar-ProgressBar-progress-bar > div {
  background-color: var(--jp-brand-color2);
  background-image: linear-gradient(
    -45deg,
    rgba(255, 255, 255, 0.2) 25%,
    transparent 25%,
    transparent 50%,
    rgba(255, 255, 255, 0.2) 50%,
    rgba(255, 255, 255, 0.2) 75%,
    transparent 75%,
    transparent
  );
  background-size: 40px 40px;
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 14px;
  color: #fff;
  text-align: center;
  animation: jp-Statusbar-ExecutionTime-progress-bar 2s linear infinite;
}

.jp-Statusbar-ProgressBar-progress-bar p {
  color: var(--jp-ui-font-color1);
  font-family: var(--jp-ui-font-family);
  font-size: var(--jp-ui-font-size1);
  line-height: 10px;
  width: 100px;
}

@keyframes jp-Statusbar-ExecutionTime-progress-bar {
  0% {
    background-position: 0 0;
  }

  100% {
    background-position: 40px 40px;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-commandpalette-search-height: 28px;
}

/*-----------------------------------------------------------------------------
| Overall styles
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  padding-bottom: 0;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Modal variant
|----------------------------------------------------------------------------*/

.jp-ModalCommandPalette {
  position: absolute;
  z-index: 10000;
  top: 38px;
  left: 30%;
  margin: 0;
  padding: 4px;
  width: 40%;
  box-shadow: var(--jp-elevation-z4);
  border-radius: 4px;
  background: var(--jp-layout-color0);
}

.jp-ModalCommandPalette .lm-CommandPalette {
  max-height: 40vh;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-close-icon::after {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-header {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-item {
  margin-left: 4px;
  margin-right: 4px;
}

.jp-ModalCommandPalette
  .lm-CommandPalette
  .lm-CommandPalette-item.lm-mod-disabled {
  display: none;
}

/*-----------------------------------------------------------------------------
| Search
|----------------------------------------------------------------------------*/

.lm-CommandPalette-search {
  padding: 4px;
  background-color: var(--jp-layout-color1);
  z-index: 2;
}

.lm-CommandPalette-wrapper {
  overflow: overlay;
  padding: 0 9px;
  background-color: var(--jp-input-active-background);
  height: 30px;
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.lm-CommandPalette.lm-mod-focused .lm-CommandPalette-wrapper {
  box-shadow: inset 0 0 0 1px var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-SearchIconGroup {
  color: white;
  background-color: var(--jp-brand-color1);
  position: absolute;
  top: 4px;
  right: 4px;
  padding: 5px 5px 1px;
}

.jp-SearchIconGroup svg {
  height: 20px;
  width: 20px;
}

.jp-SearchIconGroup .jp-icon3[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-input {
  background: transparent;
  width: calc(100% - 18px);
  float: left;
  border: none;
  outline: none;
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  line-height: var(--jp-private-commandpalette-search-height);
}

.lm-CommandPalette-input::-webkit-input-placeholder,
.lm-CommandPalette-input::-moz-placeholder,
.lm-CommandPalette-input:-ms-input-placeholder {
  color: var(--jp-ui-font-color2);
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Results
|----------------------------------------------------------------------------*/

.lm-CommandPalette-header:first-child {
  margin-top: 0;
}

.lm-CommandPalette-header {
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin-top: 8px;
  padding: 8px 0 8px 12px;
  text-transform: uppercase;
}

.lm-CommandPalette-header.lm-mod-active {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-header > mark {
  background-color: transparent;
  font-weight: bold;
  color: var(--jp-ui-font-color1);
}

.lm-CommandPalette-item {
  padding: 4px 12px 4px 4px;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  font-weight: 400;
  display: flex;
}

.lm-CommandPalette-item.lm-mod-disabled {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item.lm-mod-active {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item.lm-mod-active .lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-inverse-font-color0);
}

.lm-CommandPalette-item.lm-mod-active .jp-icon-selectable[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-item.lm-mod-active:hover:not(.lm-mod-disabled) {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item:hover:not(.lm-mod-active):not(.lm-mod-disabled) {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-itemContent {
  overflow: hidden;
}

.lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.lm-CommandPalette-item.lm-mod-disabled mark {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item .lm-CommandPalette-itemIcon {
  margin: 0 4px 0 0;
  position: relative;
  width: 16px;
  top: 2px;
  flex: 0 0 auto;
}

.lm-CommandPalette-item.lm-mod-disabled .lm-CommandPalette-itemIcon {
  opacity: 0.6;
}

.lm-CommandPalette-item .lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemCaption {
  display: none;
}

.lm-CommandPalette-content {
  background-color: var(--jp-layout-color1);
}

.lm-CommandPalette-content:empty::after {
  content: 'No results';
  margin: auto;
  margin-top: 20px;
  width: 100px;
  display: block;
  font-size: var(--jp-ui-font-size2);
  font-family: var(--jp-ui-font-family);
  font-weight: lighter;
}

.lm-CommandPalette-emptyMessage {
  text-align: center;
  margin-top: 24px;
  line-height: 1.32;
  padding: 0 8px;
  color: var(--jp-content-font-color3);
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Dialog {
  position: absolute;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  top: 0;
  left: 0;
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-dialog-background);
}

.jp-Dialog-content {
  display: flex;
  flex-direction: column;
  margin-left: auto;
  margin-right: auto;
  background: var(--jp-layout-color1);
  padding: 24px 24px 12px;
  min-width: 300px;
  min-height: 150px;
  max-width: 1000px;
  max-height: 500px;
  box-sizing: border-box;
  box-shadow: var(--jp-elevation-z20);
  word-wrap: break-word;
  border-radius: var(--jp-border-radius);

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color1);
  resize: both;
}

.jp-Dialog-content.jp-Dialog-content-small {
  max-width: 500px;
}

.jp-Dialog-button {
  overflow: visible;
}

button.jp-Dialog-button:focus {
  outline: 1px solid var(--jp-brand-color1);
  outline-offset: 4px;
  -moz-outline-radius: 0;
}

button.jp-Dialog-button:focus::-moz-focus-inner {
  border: 0;
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-accept:focus,
button.jp-Dialog-button.jp-mod-styled.jp-mod-warn:focus,
button.jp-Dialog-button.jp-mod-styled.jp-mod-reject:focus {
  outline-offset: 4px;
  -moz-outline-radius: 0;
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-accept:focus {
  outline: 1px solid var(--jp-accept-color-normal, var(--jp-brand-color1));
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-warn:focus {
  outline: 1px solid var(--jp-warn-color-normal, var(--jp-error-color1));
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-reject:focus {
  outline: 1px solid var(--jp-reject-color-normal, var(--md-grey-600));
}

button.jp-Dialog-close-button {
  padding: 0;
  height: 100%;
  min-width: unset;
  min-height: unset;
}

.jp-Dialog-header {
  display: flex;
  justify-content: space-between;
  flex: 0 0 auto;
  padding-bottom: 12px;
  font-size: var(--jp-ui-font-size3);
  font-weight: 400;
  color: var(--jp-ui-font-color1);
}

.jp-Dialog-body {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  font-size: var(--jp-ui-font-size1);
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

.jp-Dialog-footer {
  display: flex;
  flex-direction: row;
  justify-content: flex-end;
  align-items: center;
  flex: 0 0 auto;
  margin-left: -12px;
  margin-right: -12px;
  padding: 12px;
}

.jp-Dialog-checkbox {
  padding-right: 5px;
}

.jp-Dialog-checkbox > input:focus-visible {
  outline: 1px solid var(--jp-input-active-border-color);
  outline-offset: 1px;
}

.jp-Dialog-spacer {
  flex: 1 1 auto;
}

.jp-Dialog-title {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.jp-Dialog-body > .jp-select-wrapper {
  width: 100%;
}

.jp-Dialog-body > button {
  padding: 0 16px;
}

.jp-Dialog-body > label {
  line-height: 1.4;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-button.jp-mod-styled:not(:last-child) {
  margin-right: 12px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-Input-Boolean-Dialog {
  flex-direction: row-reverse;
  align-items: end;
  width: 100%;
}

.jp-Input-Boolean-Dialog > label {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MainAreaWidget > :focus {
  outline: none;
}

.jp-MainAreaWidget .jp-MainAreaWidget-error {
  padding: 6px;
}

.jp-MainAreaWidget .jp-MainAreaWidget-error > pre {
  width: auto;
  padding: 10px;
  background: var(--jp-error-color3);
  border: var(--jp-border-width) solid var(--jp-error-color1);
  border-radius: var(--jp-border-radius);
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  white-space: pre-wrap;
  word-wrap: break-word;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/**
 * google-material-color v1.2.6
 * https://github.com/danlevan/google-material-color
 */
:root {
  --md-red-50: #ffebee;
  --md-red-100: #ffcdd2;
  --md-red-200: #ef9a9a;
  --md-red-300: #e57373;
  --md-red-400: #ef5350;
  --md-red-500: #f44336;
  --md-red-600: #e53935;
  --md-red-700: #d32f2f;
  --md-red-800: #c62828;
  --md-red-900: #b71c1c;
  --md-red-A100: #ff8a80;
  --md-red-A200: #ff5252;
  --md-red-A400: #ff1744;
  --md-red-A700: #d50000;
  --md-pink-50: #fce4ec;
  --md-pink-100: #f8bbd0;
  --md-pink-200: #f48fb1;
  --md-pink-300: #f06292;
  --md-pink-400: #ec407a;
  --md-pink-500: #e91e63;
  --md-pink-600: #d81b60;
  --md-pink-700: #c2185b;
  --md-pink-800: #ad1457;
  --md-pink-900: #880e4f;
  --md-pink-A100: #ff80ab;
  --md-pink-A200: #ff4081;
  --md-pink-A400: #f50057;
  --md-pink-A700: #c51162;
  --md-purple-50: #f3e5f5;
  --md-purple-100: #e1bee7;
  --md-purple-200: #ce93d8;
  --md-purple-300: #ba68c8;
  --md-purple-400: #ab47bc;
  --md-purple-500: #9c27b0;
  --md-purple-600: #8e24aa;
  --md-purple-700: #7b1fa2;
  --md-purple-800: #6a1b9a;
  --md-purple-900: #4a148c;
  --md-purple-A100: #ea80fc;
  --md-purple-A200: #e040fb;
  --md-purple-A400: #d500f9;
  --md-purple-A700: #a0f;
  --md-deep-purple-50: #ede7f6;
  --md-deep-purple-100: #d1c4e9;
  --md-deep-purple-200: #b39ddb;
  --md-deep-purple-300: #9575cd;
  --md-deep-purple-400: #7e57c2;
  --md-deep-purple-500: #673ab7;
  --md-deep-purple-600: #5e35b1;
  --md-deep-purple-700: #512da8;
  --md-deep-purple-800: #4527a0;
  --md-deep-purple-900: #311b92;
  --md-deep-purple-A100: #b388ff;
  --md-deep-purple-A200: #7c4dff;
  --md-deep-purple-A400: #651fff;
  --md-deep-purple-A700: #6200ea;
  --md-indigo-50: #e8eaf6;
  --md-indigo-100: #c5cae9;
  --md-indigo-200: #9fa8da;
  --md-indigo-300: #7986cb;
  --md-indigo-400: #5c6bc0;
  --md-indigo-500: #3f51b5;
  --md-indigo-600: #3949ab;
  --md-indigo-700: #303f9f;
  --md-indigo-800: #283593;
  --md-indigo-900: #1a237e;
  --md-indigo-A100: #8c9eff;
  --md-indigo-A200: #536dfe;
  --md-indigo-A400: #3d5afe;
  --md-indigo-A700: #304ffe;
  --md-blue-50: #e3f2fd;
  --md-blue-100: #bbdefb;
  --md-blue-200: #90caf9;
  --md-blue-300: #64b5f6;
  --md-blue-400: #42a5f5;
  --md-blue-500: #2196f3;
  --md-blue-600: #1e88e5;
  --md-blue-700: #1976d2;
  --md-blue-800: #1565c0;
  --md-blue-900: #0d47a1;
  --md-blue-A100: #82b1ff;
  --md-blue-A200: #448aff;
  --md-blue-A400: #2979ff;
  --md-blue-A700: #2962ff;
  --md-light-blue-50: #e1f5fe;
  --md-light-blue-100: #b3e5fc;
  --md-light-blue-200: #81d4fa;
  --md-light-blue-300: #4fc3f7;
  --md-light-blue-400: #29b6f6;
  --md-light-blue-500: #03a9f4;
  --md-light-blue-600: #039be5;
  --md-light-blue-700: #0288d1;
  --md-light-blue-800: #0277bd;
  --md-light-blue-900: #01579b;
  --md-light-blue-A100: #80d8ff;
  --md-light-blue-A200: #40c4ff;
  --md-light-blue-A400: #00b0ff;
  --md-light-blue-A700: #0091ea;
  --md-cyan-50: #e0f7fa;
  --md-cyan-100: #b2ebf2;
  --md-cyan-200: #80deea;
  --md-cyan-300: #4dd0e1;
  --md-cyan-400: #26c6da;
  --md-cyan-500: #00bcd4;
  --md-cyan-600: #00acc1;
  --md-cyan-700: #0097a7;
  --md-cyan-800: #00838f;
  --md-cyan-900: #006064;
  --md-cyan-A100: #84ffff;
  --md-cyan-A200: #18ffff;
  --md-cyan-A400: #00e5ff;
  --md-cyan-A700: #00b8d4;
  --md-teal-50: #e0f2f1;
  --md-teal-100: #b2dfdb;
  --md-teal-200: #80cbc4;
  --md-teal-300: #4db6ac;
  --md-teal-400: #26a69a;
  --md-teal-500: #009688;
  --md-teal-600: #00897b;
  --md-teal-700: #00796b;
  --md-teal-800: #00695c;
  --md-teal-900: #004d40;
  --md-teal-A100: #a7ffeb;
  --md-teal-A200: #64ffda;
  --md-teal-A400: #1de9b6;
  --md-teal-A700: #00bfa5;
  --md-green-50: #e8f5e9;
  --md-green-100: #c8e6c9;
  --md-green-200: #a5d6a7;
  --md-green-300: #81c784;
  --md-green-400: #66bb6a;
  --md-green-500: #4caf50;
  --md-green-600: #43a047;
  --md-green-700: #388e3c;
  --md-green-800: #2e7d32;
  --md-green-900: #1b5e20;
  --md-green-A100: #b9f6ca;
  --md-green-A200: #69f0ae;
  --md-green-A400: #00e676;
  --md-green-A700: #00c853;
  --md-light-green-50: #f1f8e9;
  --md-light-green-100: #dcedc8;
  --md-light-green-200: #c5e1a5;
  --md-light-green-300: #aed581;
  --md-light-green-400: #9ccc65;
  --md-light-green-500: #8bc34a;
  --md-light-green-600: #7cb342;
  --md-light-green-700: #689f38;
  --md-light-green-800: #558b2f;
  --md-light-green-900: #33691e;
  --md-light-green-A100: #ccff90;
  --md-light-green-A200: #b2ff59;
  --md-light-green-A400: #76ff03;
  --md-light-green-A700: #64dd17;
  --md-lime-50: #f9fbe7;
  --md-lime-100: #f0f4c3;
  --md-lime-200: #e6ee9c;
  --md-lime-300: #dce775;
  --md-lime-400: #d4e157;
  --md-lime-500: #cddc39;
  --md-lime-600: #c0ca33;
  --md-lime-700: #afb42b;
  --md-lime-800: #9e9d24;
  --md-lime-900: #827717;
  --md-lime-A100: #f4ff81;
  --md-lime-A200: #eeff41;
  --md-lime-A400: #c6ff00;
  --md-lime-A700: #aeea00;
  --md-yellow-50: #fffde7;
  --md-yellow-100: #fff9c4;
  --md-yellow-200: #fff59d;
  --md-yellow-300: #fff176;
  --md-yellow-400: #ffee58;
  --md-yellow-500: #ffeb3b;
  --md-yellow-600: #fdd835;
  --md-yellow-700: #fbc02d;
  --md-yellow-800: #f9a825;
  --md-yellow-900: #f57f17;
  --md-yellow-A100: #ffff8d;
  --md-yellow-A200: #ff0;
  --md-yellow-A400: #ffea00;
  --md-yellow-A700: #ffd600;
  --md-amber-50: #fff8e1;
  --md-amber-100: #ffecb3;
  --md-amber-200: #ffe082;
  --md-amber-300: #ffd54f;
  --md-amber-400: #ffca28;
  --md-amber-500: #ffc107;
  --md-amber-600: #ffb300;
  --md-amber-700: #ffa000;
  --md-amber-800: #ff8f00;
  --md-amber-900: #ff6f00;
  --md-amber-A100: #ffe57f;
  --md-amber-A200: #ffd740;
  --md-amber-A400: #ffc400;
  --md-amber-A700: #ffab00;
  --md-orange-50: #fff3e0;
  --md-orange-100: #ffe0b2;
  --md-orange-200: #ffcc80;
  --md-orange-300: #ffb74d;
  --md-orange-400: #ffa726;
  --md-orange-500: #ff9800;
  --md-orange-600: #fb8c00;
  --md-orange-700: #f57c00;
  --md-orange-800: #ef6c00;
  --md-orange-900: #e65100;
  --md-orange-A100: #ffd180;
  --md-orange-A200: #ffab40;
  --md-orange-A400: #ff9100;
  --md-orange-A700: #ff6d00;
  --md-deep-orange-50: #fbe9e7;
  --md-deep-orange-100: #ffccbc;
  --md-deep-orange-200: #ffab91;
  --md-deep-orange-300: #ff8a65;
  --md-deep-orange-400: #ff7043;
  --md-deep-orange-500: #ff5722;
  --md-deep-orange-600: #f4511e;
  --md-deep-orange-700: #e64a19;
  --md-deep-orange-800: #d84315;
  --md-deep-orange-900: #bf360c;
  --md-deep-orange-A100: #ff9e80;
  --md-deep-orange-A200: #ff6e40;
  --md-deep-orange-A400: #ff3d00;
  --md-deep-orange-A700: #dd2c00;
  --md-brown-50: #efebe9;
  --md-brown-100: #d7ccc8;
  --md-brown-200: #bcaaa4;
  --md-brown-300: #a1887f;
  --md-brown-400: #8d6e63;
  --md-brown-500: #795548;
  --md-brown-600: #6d4c41;
  --md-brown-700: #5d4037;
  --md-brown-800: #4e342e;
  --md-brown-900: #3e2723;
  --md-grey-50: #fafafa;
  --md-grey-100: #f5f5f5;
  --md-grey-200: #eee;
  --md-grey-300: #e0e0e0;
  --md-grey-400: #bdbdbd;
  --md-grey-500: #9e9e9e;
  --md-grey-600: #757575;
  --md-grey-700: #616161;
  --md-grey-800: #424242;
  --md-grey-900: #212121;
  --md-blue-grey-50: #eceff1;
  --md-blue-grey-100: #cfd8dc;
  --md-blue-grey-200: #b0bec5;
  --md-blue-grey-300: #90a4ae;
  --md-blue-grey-400: #78909c;
  --md-blue-grey-500: #607d8b;
  --md-blue-grey-600: #546e7a;
  --md-blue-grey-700: #455a64;
  --md-blue-grey-800: #37474f;
  --md-blue-grey-900: #263238;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| RenderedText
|----------------------------------------------------------------------------*/

:root {
  /* This is the padding value to fill the gaps between lines containing spans with background color. */
  --jp-private-code-span-padding: calc(
    (var(--jp-code-line-height) - 1) * var(--jp-code-font-size) / 2
  );
}

.jp-RenderedText {
  text-align: left;
  padding-left: var(--jp-code-padding);
  line-height: var(--jp-code-line-height);
  font-family: var(--jp-code-font-family);
}

.jp-RenderedText pre,
.jp-RenderedJavaScript pre,
.jp-RenderedHTMLCommon pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
  border: none;
  margin: 0;
  padding: 0;
}

.jp-RenderedText pre a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedText pre a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedText pre a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* console foregrounds and backgrounds */
.jp-RenderedText pre .ansi-black-fg {
  color: #3e424d;
}

.jp-RenderedText pre .ansi-red-fg {
  color: #e75c58;
}

.jp-RenderedText pre .ansi-green-fg {
  color: #00a250;
}

.jp-RenderedText pre .ansi-yellow-fg {
  color: #ddb62b;
}

.jp-RenderedText pre .ansi-blue-fg {
  color: #208ffb;
}

.jp-RenderedText pre .ansi-magenta-fg {
  color: #d160c4;
}

.jp-RenderedText pre .ansi-cyan-fg {
  color: #60c6c8;
}

.jp-RenderedText pre .ansi-white-fg {
  color: #c5c1b4;
}

.jp-RenderedText pre .ansi-black-bg {
  background-color: #3e424d;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-red-bg {
  background-color: #e75c58;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-green-bg {
  background-color: #00a250;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-yellow-bg {
  background-color: #ddb62b;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-blue-bg {
  background-color: #208ffb;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-magenta-bg {
  background-color: #d160c4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-cyan-bg {
  background-color: #60c6c8;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-white-bg {
  background-color: #c5c1b4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-black-intense-fg {
  color: #282c36;
}

.jp-RenderedText pre .ansi-red-intense-fg {
  color: #b22b31;
}

.jp-RenderedText pre .ansi-green-intense-fg {
  color: #007427;
}

.jp-RenderedText pre .ansi-yellow-intense-fg {
  color: #b27d12;
}

.jp-RenderedText pre .ansi-blue-intense-fg {
  color: #0065ca;
}

.jp-RenderedText pre .ansi-magenta-intense-fg {
  color: #a03196;
}

.jp-RenderedText pre .ansi-cyan-intense-fg {
  color: #258f8f;
}

.jp-RenderedText pre .ansi-white-intense-fg {
  color: #a1a6b2;
}

.jp-RenderedText pre .ansi-black-intense-bg {
  background-color: #282c36;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-red-intense-bg {
  background-color: #b22b31;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-green-intense-bg {
  background-color: #007427;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-yellow-intense-bg {
  background-color: #b27d12;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-blue-intense-bg {
  background-color: #0065ca;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-magenta-intense-bg {
  background-color: #a03196;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-cyan-intense-bg {
  background-color: #258f8f;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-white-intense-bg {
  background-color: #a1a6b2;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-default-inverse-fg {
  color: var(--jp-ui-inverse-font-color0);
}

.jp-RenderedText pre .ansi-default-inverse-bg {
  background-color: var(--jp-inverse-layout-color0);
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-bold {
  font-weight: bold;
}

.jp-RenderedText pre .ansi-underline {
  text-decoration: underline;
}

.jp-RenderedText[data-mime-type='application/vnd.jupyter.stderr'] {
  background: var(--jp-rendermime-error-background);
  padding-top: var(--jp-code-padding);
}

/*-----------------------------------------------------------------------------
| RenderedLatex
|----------------------------------------------------------------------------*/

.jp-RenderedLatex {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
}

/* Left-justify outputs.*/
.jp-OutputArea-output.jp-RenderedLatex {
  padding: var(--jp-code-padding);
  text-align: left;
}

/*-----------------------------------------------------------------------------
| RenderedHTML
|----------------------------------------------------------------------------*/

.jp-RenderedHTMLCommon {
  color: var(--jp-content-font-color1);
  font-family: var(--jp-content-font-family);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);

  /* Give a bit more R padding on Markdown text to keep line lengths reasonable */
  padding-right: 20px;
}

.jp-RenderedHTMLCommon em {
  font-style: italic;
}

.jp-RenderedHTMLCommon strong {
  font-weight: bold;
}

.jp-RenderedHTMLCommon u {
  text-decoration: underline;
}

.jp-RenderedHTMLCommon a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* Headings */

.jp-RenderedHTMLCommon h1,
.jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3,
.jp-RenderedHTMLCommon h4,
.jp-RenderedHTMLCommon h5,
.jp-RenderedHTMLCommon h6 {
  line-height: var(--jp-content-heading-line-height);
  font-weight: var(--jp-content-heading-font-weight);
  font-style: normal;
  margin: var(--jp-content-heading-margin-top) 0
    var(--jp-content-heading-margin-bottom) 0;
}

.jp-RenderedHTMLCommon h1:first-child,
.jp-RenderedHTMLCommon h2:first-child,
.jp-RenderedHTMLCommon h3:first-child,
.jp-RenderedHTMLCommon h4:first-child,
.jp-RenderedHTMLCommon h5:first-child,
.jp-RenderedHTMLCommon h6:first-child {
  margin-top: calc(0.5 * var(--jp-content-heading-margin-top));
}

.jp-RenderedHTMLCommon h1:last-child,
.jp-RenderedHTMLCommon h2:last-child,
.jp-RenderedHTMLCommon h3:last-child,
.jp-RenderedHTMLCommon h4:last-child,
.jp-RenderedHTMLCommon h5:last-child,
.jp-RenderedHTMLCommon h6:last-child {
  margin-bottom: calc(0.5 * var(--jp-content-heading-margin-bottom));
}

.jp-RenderedHTMLCommon h1 {
  font-size: var(--jp-content-font-size5);
}

.jp-RenderedHTMLCommon h2 {
  font-size: var(--jp-content-font-size4);
}

.jp-RenderedHTMLCommon h3 {
  font-size: var(--jp-content-font-size3);
}

.jp-RenderedHTMLCommon h4 {
  font-size: var(--jp-content-font-size2);
}

.jp-RenderedHTMLCommon h5 {
  font-size: var(--jp-content-font-size1);
}

.jp-RenderedHTMLCommon h6 {
  font-size: var(--jp-content-font-size0);
}

/* Lists */

/* stylelint-disable selector-max-type, selector-max-compound-selectors */

.jp-RenderedHTMLCommon ul:not(.list-inline),
.jp-RenderedHTMLCommon ol:not(.list-inline) {
  padding-left: 2em;
}

.jp-RenderedHTMLCommon ul {
  list-style: disc;
}

.jp-RenderedHTMLCommon ul ul {
  list-style: square;
}

.jp-RenderedHTMLCommon ul ul ul {
  list-style: circle;
}

.jp-RenderedHTMLCommon ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol ol {
  list-style: upper-alpha;
}

.jp-RenderedHTMLCommon ol ol ol {
  list-style: lower-alpha;
}

.jp-RenderedHTMLCommon ol ol ol ol {
  list-style: lower-roman;
}

.jp-RenderedHTMLCommon ol ol ol ol ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol,
.jp-RenderedHTMLCommon ul {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon ul ul,
.jp-RenderedHTMLCommon ul ol,
.jp-RenderedHTMLCommon ol ul,
.jp-RenderedHTMLCommon ol ol {
  margin-bottom: 0;
}

/* stylelint-enable selector-max-type, selector-max-compound-selectors */

.jp-RenderedHTMLCommon hr {
  color: var(--jp-border-color2);
  background-color: var(--jp-border-color1);
  margin-top: 1em;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon > pre {
  margin: 1.5em 2em;
}

.jp-RenderedHTMLCommon pre,
.jp-RenderedHTMLCommon code {
  border: 0;
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  line-height: var(--jp-code-line-height);
  padding: 0;
  white-space: pre-wrap;
}

.jp-RenderedHTMLCommon :not(pre) > code {
  background-color: var(--jp-layout-color2);
  padding: 1px 5px;
}

/* Tables */

.jp-RenderedHTMLCommon table {
  border-collapse: collapse;
  border-spacing: 0;
  border: none;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  table-layout: fixed;
  margin-left: auto;
  margin-bottom: 1em;
  margin-right: auto;
}

.jp-RenderedHTMLCommon thead {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  vertical-align: bottom;
}

.jp-RenderedHTMLCommon td,
.jp-RenderedHTMLCommon th,
.jp-RenderedHTMLCommon tr {
  vertical-align: middle;
  padding: 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}

.jp-RenderedMarkdown.jp-RenderedHTMLCommon td,
.jp-RenderedMarkdown.jp-RenderedHTMLCommon th {
  max-width: none;
}

:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon td,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon th,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon tr {
  text-align: right;
}

.jp-RenderedHTMLCommon th {
  font-weight: bold;
}

.jp-RenderedHTMLCommon tbody tr:nth-child(odd) {
  background: var(--jp-layout-color0);
}

.jp-RenderedHTMLCommon tbody tr:nth-child(even) {
  background: var(--jp-rendermime-table-row-background);
}

.jp-RenderedHTMLCommon tbody tr:hover {
  background: var(--jp-rendermime-table-row-hover-background);
}

.jp-RenderedHTMLCommon p {
  text-align: left;
  margin: 0;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon img {
  -moz-force-broken-image-icon: 1;
}

/* Restrict to direct children as other images could be nested in other content. */
.jp-RenderedHTMLCommon > img {
  display: block;
  margin-left: 0;
  margin-right: 0;
  margin-bottom: 1em;
}

/* Change color behind transparent images if they need it... */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-light-background {
  background-color: var(--jp-inverse-layout-color1);
}

[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-dark-background {
  background-color: var(--jp-inverse-layout-color1);
}

.jp-RenderedHTMLCommon img,
.jp-RenderedImage img,
.jp-RenderedHTMLCommon svg,
.jp-RenderedSVG svg {
  max-width: 100%;
  height: auto;
}

.jp-RenderedHTMLCommon img.jp-mod-unconfined,
.jp-RenderedImage img.jp-mod-unconfined,
.jp-RenderedHTMLCommon svg.jp-mod-unconfined,
.jp-RenderedSVG svg.jp-mod-unconfined {
  max-width: none;
}

.jp-RenderedHTMLCommon .alert {
  padding: var(--jp-notebook-padding);
  border: var(--jp-border-width) solid transparent;
  border-radius: var(--jp-border-radius);
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon .alert-info {
  color: var(--jp-info-color0);
  background-color: var(--jp-info-color3);
  border-color: var(--jp-info-color2);
}

.jp-RenderedHTMLCommon .alert-info hr {
  border-color: var(--jp-info-color3);
}

.jp-RenderedHTMLCommon .alert-info > p:last-child,
.jp-RenderedHTMLCommon .alert-info > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-warning {
  color: var(--jp-warn-color0);
  background-color: var(--jp-warn-color3);
  border-color: var(--jp-warn-color2);
}

.jp-RenderedHTMLCommon .alert-warning hr {
  border-color: var(--jp-warn-color3);
}

.jp-RenderedHTMLCommon .alert-warning > p:last-child,
.jp-RenderedHTMLCommon .alert-warning > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-success {
  color: var(--jp-success-color0);
  background-color: var(--jp-success-color3);
  border-color: var(--jp-success-color2);
}

.jp-RenderedHTMLCommon .alert-success hr {
  border-color: var(--jp-success-color3);
}

.jp-RenderedHTMLCommon .alert-success > p:last-child,
.jp-RenderedHTMLCommon .alert-success > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-danger {
  color: var(--jp-error-color0);
  background-color: var(--jp-error-color3);
  border-color: var(--jp-error-color2);
}

.jp-RenderedHTMLCommon .alert-danger hr {
  border-color: var(--jp-error-color3);
}

.jp-RenderedHTMLCommon .alert-danger > p:last-child,
.jp-RenderedHTMLCommon .alert-danger > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon blockquote {
  margin: 1em 2em;
  padding: 0 1em;
  border-left: 5px solid var(--jp-border-color2);
}

a.jp-InternalAnchorLink {
  visibility: hidden;
  margin-left: 8px;
  color: var(--md-blue-800);
}

h1:hover .jp-InternalAnchorLink,
h2:hover .jp-InternalAnchorLink,
h3:hover .jp-InternalAnchorLink,
h4:hover .jp-InternalAnchorLink,
h5:hover .jp-InternalAnchorLink,
h6:hover .jp-InternalAnchorLink {
  visibility: visible;
}

.jp-RenderedHTMLCommon kbd {
  background-color: var(--jp-rendermime-table-row-background);
  border: 1px solid var(--jp-border-color0);
  border-bottom-color: var(--jp-border-color2);
  border-radius: 3px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
  display: inline-block;
  font-size: var(--jp-ui-font-size0);
  line-height: 1em;
  padding: 0.2em 0.5em;
}

/* Most direct children of .jp-RenderedHTMLCommon have a margin-bottom of 1.0.
 * At the bottom of cells this is a bit too much as there is also spacing
 * between cells. Going all the way to 0 gets too tight between markdown and
 * code cells.
 */
.jp-RenderedHTMLCommon > *:last-child {
  margin-bottom: 0.5em;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-cursor-backdrop {
  position: fixed;
  width: 200px;
  height: 200px;
  margin-top: -100px;
  margin-left: -100px;
  will-change: transform;
  z-index: 100;
}

.lm-mod-drag-image {
  will-change: transform;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-lineFormSearch {
  padding: 4px 12px;
  background-color: var(--jp-layout-color2);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
  font-size: var(--jp-ui-font-size1);
}

.jp-lineFormCaption {
  font-size: var(--jp-ui-font-size0);
  line-height: var(--jp-ui-font-size1);
  margin-top: 4px;
  color: var(--jp-ui-font-color0);
}

.jp-baseLineForm {
  border: none;
  border-radius: 0;
  position: absolute;
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  outline: none;
}

.jp-lineFormButtonContainer {
  top: 4px;
  right: 8px;
  height: 24px;
  padding: 0 12px;
  width: 12px;
}

.jp-lineFormButtonIcon {
  top: 0;
  right: 0;
  background-color: var(--jp-brand-color1);
  height: 100%;
  width: 100%;
  box-sizing: border-box;
  padding: 4px 6px;
}

.jp-lineFormButton {
  top: 0;
  right: 0;
  background-color: transparent;
  height: 100%;
  width: 100%;
  box-sizing: border-box;
}

.jp-lineFormWrapper {
  overflow: hidden;
  padding: 0 8px;
  border: 1px solid var(--jp-border-color0);
  background-color: var(--jp-input-active-background);
  height: 22px;
}

.jp-lineFormWrapperFocusWithin {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-lineFormInput {
  background: transparent;
  width: 200px;
  height: 100%;
  border: none;
  outline: none;
  color: var(--jp-ui-font-color0);
  line-height: 28px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-JSONEditor {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.jp-JSONEditor-host {
  flex: 1 1 auto;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0;
  background: var(--jp-layout-color0);
  min-height: 50px;
  padding: 1px;
}

.jp-JSONEditor.jp-mod-error .jp-JSONEditor-host {
  border-color: red;
  outline-color: red;
}

.jp-JSONEditor-header {
  display: flex;
  flex: 1 0 auto;
  padding: 0 0 0 12px;
}

.jp-JSONEditor-header label {
  flex: 0 0 auto;
}

.jp-JSONEditor-commitButton {
  height: 16px;
  width: 16px;
  background-size: 18px;
  background-repeat: no-repeat;
  background-position: center;
}

.jp-JSONEditor-host.jp-mod-focused {
  background-color: var(--jp-input-active-background);
  border: 1px solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

.jp-Editor.jp-mod-dropTarget {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/
.jp-DocumentSearch-input {
  border: none;
  outline: none;
  color: var(--jp-ui-font-color0);
  font-size: var(--jp-ui-font-size1);
  background-color: var(--jp-layout-color0);
  font-family: var(--jp-ui-font-family);
  padding: 2px 1px;
  resize: none;
}

.jp-DocumentSearch-overlay {
  position: absolute;
  background-color: var(--jp-toolbar-background);
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  border-left: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  top: 0;
  right: 0;
  z-index: 7;
  min-width: 405px;
  padding: 2px;
  font-size: var(--jp-ui-font-size1);

  --jp-private-document-search-button-height: 20px;
}

.jp-DocumentSearch-overlay button {
  background-color: var(--jp-toolbar-background);
  outline: 0;
}

.jp-DocumentSearch-overlay button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-overlay button:active {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-overlay-row {
  display: flex;
  align-items: center;
  margin-bottom: 2px;
}

.jp-DocumentSearch-button-content {
  display: inline-block;
  cursor: pointer;
  box-sizing: border-box;
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-button-content svg {
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-input-wrapper {
  border: var(--jp-border-width) solid var(--jp-border-color0);
  display: flex;
  background-color: var(--jp-layout-color0);
  margin: 2px;
}

.jp-DocumentSearch-input-wrapper:focus-within {
  border-color: var(--jp-cell-editor-active-border-color);
}

.jp-DocumentSearch-toggle-wrapper,
.jp-DocumentSearch-button-wrapper {
  all: initial;
  overflow: hidden;
  display: inline-block;
  border: none;
  box-sizing: border-box;
}

.jp-DocumentSearch-toggle-wrapper {
  width: 14px;
  height: 14px;
}

.jp-DocumentSearch-button-wrapper {
  width: var(--jp-private-document-search-button-height);
  height: var(--jp-private-document-search-button-height);
}

.jp-DocumentSearch-toggle-wrapper:focus,
.jp-DocumentSearch-button-wrapper:focus {
  outline: var(--jp-border-width) solid
    var(--jp-cell-editor-active-border-color);
  outline-offset: -1px;
}

.jp-DocumentSearch-toggle-wrapper,
.jp-DocumentSearch-button-wrapper,
.jp-DocumentSearch-button-content:focus {
  outline: none;
}

.jp-DocumentSearch-toggle-placeholder {
  width: 5px;
}

.jp-DocumentSearch-input-button::before {
  display: block;
  padding-top: 100%;
}

.jp-DocumentSearch-input-button-off {
  opacity: var(--jp-search-toggle-off-opacity);
}

.jp-DocumentSearch-input-button-off:hover {
  opacity: var(--jp-search-toggle-hover-opacity);
}

.jp-DocumentSearch-input-button-on {
  opacity: var(--jp-search-toggle-on-opacity);
}

.jp-DocumentSearch-index-counter {
  padding-left: 10px;
  padding-right: 10px;
  user-select: none;
  min-width: 35px;
  display: inline-block;
}

.jp-DocumentSearch-up-down-wrapper {
  display: inline-block;
  padding-right: 2px;
  margin-left: auto;
  white-space: nowrap;
}

.jp-DocumentSearch-spacer {
  margin-left: auto;
}

.jp-DocumentSearch-up-down-wrapper button {
  outline: 0;
  border: none;
  width: var(--jp-private-document-search-button-height);
  height: var(--jp-private-document-search-button-height);
  vertical-align: middle;
  margin: 1px 5px 2px;
}

.jp-DocumentSearch-up-down-button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-up-down-button:active {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-filter-button {
  border-radius: var(--jp-border-radius);
}

.jp-DocumentSearch-filter-button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-filter-button-enabled {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-filter-button-enabled:hover {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-search-options {
  padding: 0 8px;
  margin-left: 3px;
  width: 100%;
  display: grid;
  justify-content: start;
  grid-template-columns: 1fr 1fr;
  align-items: center;
  justify-items: stretch;
}

.jp-DocumentSearch-search-filter-disabled {
  color: var(--jp-ui-font-color2);
}

.jp-DocumentSearch-search-filter {
  display: flex;
  align-items: center;
  user-select: none;
}

.jp-DocumentSearch-regex-error {
  color: var(--jp-error-color0);
}

.jp-DocumentSearch-replace-button-wrapper {
  overflow: hidden;
  display: inline-block;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color0);
  margin: auto 2px;
  padding: 1px 4px;
  height: calc(var(--jp-private-document-search-button-height) + 2px);
}

.jp-DocumentSearch-replace-button-wrapper:focus {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
}

.jp-DocumentSearch-replace-button {
  display: inline-block;
  text-align: center;
  cursor: pointer;
  box-sizing: border-box;
  color: var(--jp-ui-font-color1);

  /* height - 2 * (padding of wrapper) */
  line-height: calc(var(--jp-private-document-search-button-height) - 2px);
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-replace-button:focus {
  outline: none;
}

.jp-DocumentSearch-replace-wrapper-class {
  margin-left: 14px;
  display: flex;
}

.jp-DocumentSearch-replace-toggle {
  border: none;
  background-color: var(--jp-toolbar-background);
  border-radius: var(--jp-border-radius);
}

.jp-DocumentSearch-replace-toggle:hover {
  background-color: var(--jp-layout-color2);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.cm-editor {
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  border: 0;
  border-radius: 0;
  height: auto;

  /* Changed to auto to autogrow */
}

.cm-editor pre {
  padding: 0 var(--jp-code-padding);
}

.jp-CodeMirrorEditor[data-type='inline'] .cm-dialog {
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

.jp-CodeMirrorEditor {
  cursor: text;
}

/* When zoomed out 67% and 33% on a screen of 1440 width x 900 height */
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .jp-CodeMirrorEditor[data-type='inline'] .cm-cursor {
    border-left: var(--jp-code-cursor-width1) solid
      var(--jp-editor-cursor-color);
  }
}

/* When zoomed out less than 33% */
@media screen and (min-width: 4320px) {
  .jp-CodeMirrorEditor[data-type='inline'] .cm-cursor {
    border-left: var(--jp-code-cursor-width2) solid
      var(--jp-editor-cursor-color);
  }
}

.cm-editor.jp-mod-readOnly .cm-cursor {
  display: none;
}

.jp-CollaboratorCursor {
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: none;
  border-bottom: 3px solid;
  background-clip: content-box;
  margin-left: -5px;
  margin-right: -5px;
}

.cm-searching,
.cm-searching span {
  /* `.cm-searching span`: we need to override syntax highlighting */
  background-color: var(--jp-search-unselected-match-background-color);
  color: var(--jp-search-unselected-match-color);
}

.cm-searching::selection,
.cm-searching span::selection {
  background-color: var(--jp-search-unselected-match-background-color);
  color: var(--jp-search-unselected-match-color);
}

.jp-current-match > .cm-searching,
.jp-current-match > .cm-searching span,
.cm-searching > .jp-current-match,
.cm-searching > .jp-current-match span {
  background-color: var(--jp-search-selected-match-background-color);
  color: var(--jp-search-selected-match-color);
}

.jp-current-match > .cm-searching::selection,
.cm-searching > .jp-current-match::selection,
.jp-current-match > .cm-searching span::selection {
  background-color: var(--jp-search-selected-match-background-color);
  color: var(--jp-search-selected-match-color);
}

.cm-trailingspace {
  background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAsElEQVQIHQGlAFr/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA7+r3zKmT0/+pk9P/7+r3zAAAAAAAAAAABAAAAAAAAAAA6OPzM+/q9wAAAAAA6OPzMwAAAAAAAAAAAgAAAAAAAAAAGR8NiRQaCgAZIA0AGR8NiQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQyoYJ/SY80UAAAAASUVORK5CYII=);
  background-position: center left;
  background-repeat: repeat-x;
}

.jp-CollaboratorCursor-hover {
  position: absolute;
  z-index: 1;
  transform: translateX(-50%);
  color: white;
  border-radius: 3px;
  padding-left: 4px;
  padding-right: 4px;
  padding-top: 1px;
  padding-bottom: 1px;
  text-align: center;
  font-size: var(--jp-ui-font-size1);
  white-space: nowrap;
}

.jp-CodeMirror-ruler {
  border-left: 1px dashed var(--jp-border-color2);
}

/* Styles for shared cursors (remote cursor locations and selected ranges) */
.jp-CodeMirrorEditor .cm-ySelectionCaret {
  position: relative;
  border-left: 1px solid black;
  margin-left: -1px;
  margin-right: -1px;
  box-sizing: border-box;
}

.jp-CodeMirrorEditor .cm-ySelectionCaret > .cm-ySelectionInfo {
  white-space: nowrap;
  position: absolute;
  top: -1.15em;
  padding-bottom: 0.05em;
  left: -1px;
  font-size: 0.95em;
  font-family: var(--jp-ui-font-family);
  font-weight: bold;
  line-height: normal;
  user-select: none;
  color: white;
  padding-left: 2px;
  padding-right: 2px;
  z-index: 101;
  transition: opacity 0.3s ease-in-out;
}

.jp-CodeMirrorEditor .cm-ySelectionInfo {
  transition-delay: 0.7s;
  opacity: 0;
}

.jp-CodeMirrorEditor .cm-ySelectionCaret:hover > .cm-ySelectionInfo {
  opacity: 1;
  transition-delay: 0s;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MimeDocument {
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-filebrowser-button-height: 28px;
  --jp-private-filebrowser-button-width: 48px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FileBrowser .jp-SidePanel-content {
  display: flex;
  flex-direction: column;
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  flex-wrap: wrap;
  row-gap: 12px;
  border-bottom: none;
  height: auto;
  margin: 8px 12px 0;
  box-shadow: none;
  padding: 0;
  justify-content: flex-start;
}

.jp-FileBrowser-Panel {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
}

.jp-BreadCrumbs {
  flex: 0 0 auto;
  margin: 8px 12px;
}

.jp-BreadCrumbs-item {
  margin: 0 2px;
  padding: 0 2px;
  border-radius: var(--jp-border-radius);
  cursor: pointer;
}

.jp-BreadCrumbs-item:hover {
  background-color: var(--jp-layout-color2);
}

.jp-BreadCrumbs-item:first-child {
  margin-left: 0;
}

.jp-BreadCrumbs-item.jp-mod-dropTarget {
  background-color: var(--jp-brand-color2);
  opacity: 0.7;
}

/*-----------------------------------------------------------------------------
| Buttons
|----------------------------------------------------------------------------*/

.jp-FileBrowser-toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  padding-left: 0;
  padding-right: 2px;
  align-items: center;
  height: unset;
}

.jp-FileBrowser-toolbar > .jp-Toolbar-item .jp-ToolbarButtonComponent {
  width: 40px;
}

/*-----------------------------------------------------------------------------
| Other styles
|----------------------------------------------------------------------------*/

.jp-FileDialog.jp-mod-conflict input {
  color: var(--jp-error-color1);
}

.jp-FileDialog .jp-new-name-title {
  margin-top: 12px;
}

.jp-LastModified-hidden {
  display: none;
}

.jp-FileSize-hidden {
  display: none;
}

.jp-FileBrowser .lm-AccordionPanel > h3:first-child {
  display: none;
}

/*-----------------------------------------------------------------------------
| DirListing
|----------------------------------------------------------------------------*/

.jp-DirListing {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  outline: 0;
}

.jp-DirListing-header {
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  align-items: center;
  overflow: hidden;
  border-top: var(--jp-border-width) solid var(--jp-border-color2);
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
}

.jp-DirListing-headerItem {
  padding: 4px 12px 2px;
  font-weight: 500;
}

.jp-DirListing-headerItem:hover {
  background: var(--jp-layout-color2);
}

.jp-DirListing-headerItem.jp-id-name {
  flex: 1 0 84px;
}

.jp-DirListing-headerItem.jp-id-modified {
  flex: 0 0 112px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-DirListing-headerItem.jp-id-filesize {
  flex: 0 0 75px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-id-narrow {
  display: none;
  flex: 0 0 5px;
  padding: 4px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
  color: var(--jp-border-color2);
}

.jp-DirListing-narrow .jp-id-narrow {
  display: block;
}

.jp-DirListing-narrow .jp-id-modified,
.jp-DirListing-narrow .jp-DirListing-itemModified {
  display: none;
}

.jp-DirListing-headerItem.jp-mod-selected {
  font-weight: 600;
}

/* increase specificity to override bundled default */
.jp-DirListing-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-content mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.jp-DirListing-content .jp-DirListing-item.jp-mod-selected mark {
  color: var(--jp-ui-inverse-font-color0);
}

/* Style the directory listing content when a user drops a file to upload */
.jp-DirListing.jp-mod-native-drop .jp-DirListing-content {
  outline: 5px dashed rgba(128, 128, 128, 0.5);
  outline-offset: -10px;
  cursor: copy;
}

.jp-DirListing-item {
  display: flex;
  flex-direction: row;
  align-items: center;
  padding: 4px 12px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-DirListing-checkboxWrapper {
  /* Increases hit area of checkbox. */
  padding: 4px;
}

.jp-DirListing-header
  .jp-DirListing-checkboxWrapper
  + .jp-DirListing-headerItem {
  padding-left: 4px;
}

.jp-DirListing-content .jp-DirListing-checkboxWrapper {
  position: relative;
  left: -4px;
  margin: -4px 0 -4px -8px;
}

.jp-DirListing-checkboxWrapper.jp-mod-visible {
  visibility: visible;
}

/* For devices that support hovering, hide checkboxes until hovered, selected...
*/
@media (hover: hover) {
  .jp-DirListing-checkboxWrapper {
    visibility: hidden;
  }

  .jp-DirListing-item:hover .jp-DirListing-checkboxWrapper,
  .jp-DirListing-item.jp-mod-selected .jp-DirListing-checkboxWrapper {
    visibility: visible;
  }
}

.jp-DirListing-item[data-is-dot] {
  opacity: 75%;
}

.jp-DirListing-item.jp-mod-selected {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.jp-DirListing-item.jp-mod-dropTarget {
  background: var(--jp-brand-color3);
}

.jp-DirListing-item:hover:not(.jp-mod-selected) {
  background: var(--jp-layout-color2);
}

.jp-DirListing-itemIcon {
  flex: 0 0 20px;
  margin-right: 4px;
}

.jp-DirListing-itemText {
  flex: 1 0 64px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  user-select: none;
}

.jp-DirListing-itemText:focus {
  outline-width: 2px;
  outline-color: var(--jp-inverse-layout-color1);
  outline-style: solid;
  outline-offset: 1px;
}

.jp-DirListing-item.jp-mod-selected .jp-DirListing-itemText:focus {
  outline-color: var(--jp-layout-color1);
}

.jp-DirListing-itemModified {
  flex: 0 0 125px;
  text-align: right;
}

.jp-DirListing-itemFileSize {
  flex: 0 0 90px;
  text-align: right;
}

.jp-DirListing-editor {
  flex: 1 0 64px;
  outline: none;
  border: none;
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-item.jp-mod-running .jp-DirListing-itemIcon::before {
  color: var(--jp-success-color1);
  content: '\25CF';
  font-size: 8px;
  position: absolute;
  left: -8px;
}

.jp-DirListing-item.jp-mod-running.jp-mod-selected
  .jp-DirListing-itemIcon::before {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-DirListing-item.lm-mod-drag-image,
.jp-DirListing-item.jp-mod-selected.lm-mod-drag-image {
  font-size: var(--jp-ui-font-size1);
  padding-left: 4px;
  margin-left: 4px;
  width: 160px;
  background-color: var(--jp-ui-inverse-font-color2);
  box-shadow: var(--jp-elevation-z2);
  border-radius: 0;
  color: var(--jp-ui-font-color1);
  transform: translateX(-40%) translateY(-58%);
}

.jp-Document {
  min-width: 120px;
  min-height: 120px;
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Main OutputArea
| OutputArea has a list of Outputs
|----------------------------------------------------------------------------*/

.jp-OutputArea {
  overflow-y: auto;
}

.jp-OutputArea-child {
  display: table;
  table-layout: fixed;
  width: 100%;
  overflow: hidden;
}

.jp-OutputPrompt {
  width: var(--jp-cell-prompt-width);
  color: var(--jp-cell-outprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);

  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-OutputArea-prompt {
  display: table-cell;
  vertical-align: top;
}

.jp-OutputArea-output {
  display: table-cell;
  width: 100%;
  height: auto;
  overflow: auto;
  user-select: text;
  -moz-user-select: text;
  -webkit-user-select: text;
  -ms-user-select: text;
}

.jp-OutputArea .jp-RenderedText {
  padding-left: 1ch;
}

/**
 * Prompt overlay.
 */

.jp-OutputArea-promptOverlay {
  position: absolute;
  top: 0;
  width: var(--jp-cell-prompt-width);
  height: 100%;
  opacity: 0.5;
}

.jp-OutputArea-promptOverlay:hover {
  background: var(--jp-layout-color2);
  box-shadow: inset 0 0 1px var(--jp-inverse-layout-color0);
  cursor: zoom-out;
}

.jp-mod-outputsScrolled .jp-OutputArea-promptOverlay:hover {
  cursor: zoom-in;
}

/**
 * Isolated output.
 */
.jp-OutputArea-output.jp-mod-isolated {
  width: 100%;
  display: block;
}

/*
When drag events occur, `lm-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated {
  position: relative;
}

body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/* pre */

.jp-OutputArea-output pre {
  border: none;
  margin: 0;
  padding: 0;
  overflow-x: auto;
  overflow-y: auto;
  word-break: break-all;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* tables */

.jp-OutputArea-output.jp-RenderedHTMLCommon table {
  margin-left: 0;
  margin-right: 0;
}

/* description lists */

.jp-OutputArea-output dl,
.jp-OutputArea-output dt,
.jp-OutputArea-output dd {
  display: block;
}

.jp-OutputArea-output dl {
  width: 100%;
  overflow: hidden;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dt {
  font-weight: bold;
  float: left;
  width: 20%;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dd {
  float: left;
  width: 80%;
  padding: 0;
  margin: 0;
}

.jp-TrimmedOutputs pre {
  background: var(--jp-layout-color3);
  font-size: calc(var(--jp-code-font-size) * 1.4);
  text-align: center;
  text-transform: uppercase;
}

/* Hide the gutter in case of
 *  - nested output areas (e.g. in the case of output widgets)
 *  - mirrored output areas
 */
.jp-OutputArea .jp-OutputArea .jp-OutputArea-prompt {
  display: none;
}

/* Hide empty lines in the output area, for instance due to cleared widgets */
.jp-OutputArea-prompt:empty {
  padding: 0;
  border: 0;
}

/*-----------------------------------------------------------------------------
| executeResult is added to any Output-result for the display of the object
| returned by a cell
|----------------------------------------------------------------------------*/

.jp-OutputArea-output.jp-OutputArea-executeResult {
  margin-left: 0;
  width: 100%;
}

/* Text output with the Out[] prompt needs a top padding to match the
 * alignment of the Out[] prompt itself.
 */
.jp-OutputArea-executeResult .jp-RenderedText.jp-OutputArea-output {
  padding-top: var(--jp-code-padding);
  border-top: var(--jp-border-width) solid transparent;
}

/*-----------------------------------------------------------------------------
| The Stdin output
|----------------------------------------------------------------------------*/

.jp-Stdin-prompt {
  color: var(--jp-content-font-color0);
  padding-right: var(--jp-code-padding);
  vertical-align: baseline;
  flex: 0 0 auto;
}

.jp-Stdin-input {
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  color: inherit;
  background-color: inherit;
  width: 42%;
  min-width: 200px;

  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;

  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0 0.25em;
  margin: 0 0.25em;
  flex: 0 0 70%;
}

.jp-Stdin-input::placeholder {
  opacity: 0;
}

.jp-Stdin-input:focus {
  box-shadow: none;
}

.jp-Stdin-input:focus::placeholder {
  opacity: 1;
}

/*-----------------------------------------------------------------------------
| Output Area View
|----------------------------------------------------------------------------*/

.jp-LinkedOutputView .jp-OutputArea {
  height: 100%;
  display: block;
}

.jp-LinkedOutputView .jp-OutputArea-output:only-child {
  height: 100%;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

@media print {
  .jp-OutputArea-child {
    break-inside: avoid-page;
  }
}

/*-----------------------------------------------------------------------------
| Mobile
|----------------------------------------------------------------------------*/
@media only screen and (max-width: 760px) {
  .jp-OutputPrompt {
    display: table-row;
    text-align: left;
  }

  .jp-OutputArea-child .jp-OutputArea-output {
    display: table-row;
    margin-left: var(--jp-notebook-padding);
  }
}

/* Trimmed outputs warning */
.jp-TrimmedOutputs > a {
  margin: 10px;
  text-decoration: none;
  cursor: pointer;
}

.jp-TrimmedOutputs > a:hover {
  text-decoration: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Table of Contents
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toc-active-width: 4px;
}

.jp-TableOfContents {
  display: flex;
  flex-direction: column;
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  height: 100%;
}

.jp-TableOfContents-placeholder {
  text-align: center;
}

.jp-TableOfContents-placeholderContent {
  color: var(--jp-content-font-color2);
  padding: 8px;
}

.jp-TableOfContents-placeholderContent > h3 {
  margin-bottom: var(--jp-content-heading-margin-bottom);
}

.jp-TableOfContents .jp-SidePanel-content {
  overflow-y: auto;
}

.jp-TableOfContents-tree {
  margin: 4px;
}

.jp-TableOfContents ol {
  list-style-type: none;
}

/* stylelint-disable-next-line selector-max-type */
.jp-TableOfContents li > ol {
  /* Align left border with triangle icon center */
  padding-left: 11px;
}

.jp-TableOfContents-content {
  /* left margin for the active heading indicator */
  margin: 0 0 0 var(--jp-private-toc-active-width);
  padding: 0;
  background-color: var(--jp-layout-color1);
}

.jp-tocItem {
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-tocItem-heading {
  display: flex;
  cursor: pointer;
}

.jp-tocItem-heading:hover {
  background-color: var(--jp-layout-color2);
}

.jp-tocItem-content {
  display: block;
  padding: 4px 0;
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow-x: hidden;
}

.jp-tocItem-collapser {
  height: 20px;
  margin: 2px 2px 0;
  padding: 0;
  background: none;
  border: none;
  cursor: pointer;
}

.jp-tocItem-collapser:hover {
  background-color: var(--jp-layout-color3);
}

/* Active heading indicator */

.jp-tocItem-heading::before {
  content: ' ';
  background: transparent;
  width: var(--jp-private-toc-active-width);
  height: 24px;
  position: absolute;
  left: 0;
  border-radius: var(--jp-border-radius);
}

.jp-tocItem-heading.jp-tocItem-active::before {
  background-color: var(--jp-brand-color1);
}

.jp-tocItem-heading:hover.jp-tocItem-active::before {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapser {
  flex: 0 0 var(--jp-cell-collapser-width);
  padding: 0;
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
  border-radius: var(--jp-border-radius);
  opacity: 1;
}

.jp-Collapser-child {
  display: block;
  width: 100%;
  box-sizing: border-box;

  /* height: 100% doesn't work because the height of its parent is computed from content */
  position: absolute;
  top: 0;
  bottom: 0;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

/*
Hiding collapsers in print mode.

Note: input and output wrappers have "display: block" propery in print mode.
*/

@media print {
  .jp-Collapser {
    display: none;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Header/Footer
|----------------------------------------------------------------------------*/

/* Hidden by zero height by default */
.jp-CellHeader,
.jp-CellFooter {
  height: 0;
  width: 100%;
  padding: 0;
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Input
|----------------------------------------------------------------------------*/

/* All input areas */
.jp-InputArea {
  display: table;
  table-layout: fixed;
  width: 100%;
  overflow: hidden;
}

.jp-InputArea-editor {
  display: table-cell;
  overflow: hidden;
  vertical-align: top;

  /* This is the non-active, default styling */
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0;
  background: var(--jp-cell-editor-background);
}

.jp-InputPrompt {
  display: table-cell;
  vertical-align: top;
  width: var(--jp-cell-prompt-width);
  color: var(--jp-cell-inprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  opacity: var(--jp-cell-prompt-opacity);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;

  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

/*-----------------------------------------------------------------------------
| Mobile
|----------------------------------------------------------------------------*/
@media only screen and (max-width: 760px) {
  .jp-InputArea-editor {
    display: table-row;
    margin-left: var(--jp-notebook-padding);
  }

  .jp-InputPrompt {
    display: table-row;
    text-align: left;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Placeholder {
  display: table;
  table-layout: fixed;
  width: 100%;
}

.jp-Placeholder-prompt {
  display: table-cell;
  box-sizing: border-box;
}

.jp-Placeholder-content {
  display: table-cell;
  padding: 4px 6px;
  border: 1px solid transparent;
  border-radius: 0;
  background: none;
  box-sizing: border-box;
  cursor: pointer;
}

.jp-Placeholder-contentContainer {
  display: flex;
}

.jp-Placeholder-content:hover,
.jp-InputPlaceholder > .jp-Placeholder-content:hover {
  border-color: var(--jp-layout-color3);
}

.jp-Placeholder-content .jp-MoreHorizIcon {
  width: 32px;
  height: 16px;
  border: 1px solid transparent;
  border-radius: var(--jp-border-radius);
}

.jp-Placeholder-content .jp-MoreHorizIcon:hover {
  border: 1px solid var(--jp-border-color1);
  box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.25);
  background-color: var(--jp-layout-color0);
}

.jp-PlaceholderText {
  white-space: nowrap;
  overflow-x: hidden;
  color: var(--jp-inverse-layout-color3);
  font-family: var(--jp-code-font-family);
}

.jp-InputPlaceholder > .jp-Placeholder-content {
  border-color: var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-cell-scrolling-output-offset: 5px;
}

/*-----------------------------------------------------------------------------
| Cell
|----------------------------------------------------------------------------*/

.jp-Cell {
  padding: var(--jp-cell-padding);
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Common input/output
|----------------------------------------------------------------------------*/

.jp-Cell-inputWrapper,
.jp-Cell-outputWrapper {
  display: flex;
  flex-direction: row;
  padding: 0;
  margin: 0;

  /* Added to reveal the box-shadow on the input and output collapsers. */
  overflow: visible;
}

/* Only input/output areas inside cells */
.jp-Cell-inputArea,
.jp-Cell-outputArea {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Collapser
|----------------------------------------------------------------------------*/

/* Make the output collapser disappear when there is not output, but do so
 * in a manner that leaves it in the layout and preserves its width.
 */
.jp-Cell.jp-mod-noOutputs .jp-Cell-outputCollapser {
  border: none !important;
  background: transparent !important;
}

.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputCollapser {
  min-height: var(--jp-cell-collapser-min-height);
}

/*-----------------------------------------------------------------------------
| Output
|----------------------------------------------------------------------------*/

/* Put a space between input and output when there IS output */
.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputWrapper {
  margin-top: 5px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea {
  overflow-y: auto;
  max-height: 24em;
  margin-left: var(--jp-private-cell-scrolling-output-offset);
  resize: vertical;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea[style*='height'] {
  max-height: unset;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea::after {
  content: ' ';
  box-shadow: inset 0 0 6px 2px rgb(0 0 0 / 30%);
  width: 100%;
  height: 100%;
  position: sticky;
  bottom: 0;
  top: 0;
  margin-top: -50%;
  float: left;
  display: block;
  pointer-events: none;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-child {
  padding-top: 6px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  width: calc(
    var(--jp-cell-prompt-width) - var(--jp-private-cell-scrolling-output-offset)
  );
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-promptOverlay {
  left: calc(-1 * var(--jp-private-cell-scrolling-output-offset));
}

/*-----------------------------------------------------------------------------
| CodeCell
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| MarkdownCell
|----------------------------------------------------------------------------*/

.jp-MarkdownOutput {
  display: table-cell;
  width: 100%;
  margin-top: 0;
  margin-bottom: 0;
  padding-left: var(--jp-code-padding);
}

.jp-MarkdownOutput.jp-RenderedHTMLCommon {
  overflow: auto;
}

/* collapseHeadingButton (show always if hiddenCellsButton is _not_ shown) */
.jp-collapseHeadingButton {
  display: flex;
  min-height: var(--jp-cell-collapser-min-height);
  font-size: var(--jp-code-font-size);
  position: absolute;
  background-color: transparent;
  background-size: 25px;
  background-repeat: no-repeat;
  background-position-x: center;
  background-position-y: top;
  background-image: var(--jp-icon-caret-down);
  right: 0;
  top: 0;
  bottom: 0;
}

.jp-collapseHeadingButton.jp-mod-collapsed {
  background-image: var(--jp-icon-caret-right);
}

/*
 set the container font size to match that of content
 so that the nested collapse buttons have the right size
*/
.jp-MarkdownCell .jp-InputPrompt {
  font-size: var(--jp-content-font-size1);
}

/*
  Align collapseHeadingButton with cell top header
  The font sizes are identical to the ones in packages/rendermime/style/base.css
*/
.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='1'] {
  font-size: var(--jp-content-font-size5);
  background-position-y: calc(0.3 * var(--jp-content-font-size5));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='2'] {
  font-size: var(--jp-content-font-size4);
  background-position-y: calc(0.3 * var(--jp-content-font-size4));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='3'] {
  font-size: var(--jp-content-font-size3);
  background-position-y: calc(0.3 * var(--jp-content-font-size3));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='4'] {
  font-size: var(--jp-content-font-size2);
  background-position-y: calc(0.3 * var(--jp-content-font-size2));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='5'] {
  font-size: var(--jp-content-font-size1);
  background-position-y: top;
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='6'] {
  font-size: var(--jp-content-font-size0);
  background-position-y: top;
}

/* collapseHeadingButton (show only on (hover,active) if hiddenCellsButton is shown) */
.jp-Notebook.jp-mod-showHiddenCellsButton .jp-collapseHeadingButton {
  display: none;
}

.jp-Notebook.jp-mod-showHiddenCellsButton
  :is(.jp-MarkdownCell:hover, .jp-mod-active)
  .jp-collapseHeadingButton {
  display: flex;
}

/* showHiddenCellsButton (only show if jp-mod-showHiddenCellsButton is set, which
is a consequence of the showHiddenCellsButton option in Notebook Settings)*/
.jp-Notebook.jp-mod-showHiddenCellsButton .jp-showHiddenCellsButton {
  margin-left: calc(var(--jp-cell-prompt-width) + 2 * var(--jp-code-padding));
  margin-top: var(--jp-code-padding);
  border: 1px solid var(--jp-border-color2);
  background-color: var(--jp-border-color3) !important;
  color: var(--jp-content-font-color0) !important;
  display: flex;
}

.jp-Notebook.jp-mod-showHiddenCellsButton .jp-showHiddenCellsButton:hover {
  background-color: var(--jp-border-color2) !important;
}

.jp-showHiddenCellsButton {
  display: none;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

/*
Using block instead of flex to allow the use of the break-inside CSS property for
cell outputs.
*/

@media print {
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-notebook-toolbar-padding: 2px 5px 2px 2px;
}

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-NotebookPanel-toolbar {
  padding: var(--jp-notebook-toolbar-padding);

  /* disable paint containment from lumino 2.0 default strict CSS containment */
  contain: style size !important;
}

.jp-Toolbar-item.jp-Notebook-toolbarCellType .jp-select-wrapper.jp-mod-focused {
  border: none;
  box-shadow: none;
}

.jp-Notebook-toolbarCellTypeDropdown select {
  height: 24px;
  font-size: var(--jp-ui-font-size1);
  line-height: 14px;
  border-radius: 0;
  display: block;
}

.jp-Notebook-toolbarCellTypeDropdown span {
  top: 5px !important;
}

.jp-Toolbar-responsive-popup {
  position: absolute;
  height: fit-content;
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  justify-content: flex-end;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: var(--jp-notebook-toolbar-padding);
  z-index: 1;
  right: 0;
  top: 0;
}

.jp-Toolbar > .jp-Toolbar-responsive-opener {
  margin-left: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-Notebook-ExecutionIndicator {
  position: relative;
  display: inline-block;
  height: 100%;
  z-index: 9997;
}

.jp-Notebook-ExecutionIndicator-tooltip {
  visibility: hidden;
  height: auto;
  width: max-content;
  width: -moz-max-content;
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color1);
  text-align: justify;
  border-radius: 6px;
  padding: 0 5px;
  position: fixed;
  display: table;
}

.jp-Notebook-ExecutionIndicator-tooltip.up {
  transform: translateX(-50%) translateY(-100%) translateY(-32px);
}

.jp-Notebook-ExecutionIndicator-tooltip.down {
  transform: translateX(calc(-100% + 16px)) translateY(5px);
}

.jp-Notebook-ExecutionIndicator-tooltip.hidden {
  display: none;
}

.jp-Notebook-ExecutionIndicator:hover .jp-Notebook-ExecutionIndicator-tooltip {
  visibility: visible;
}

.jp-Notebook-ExecutionIndicator span {
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  color: var(--jp-ui-font-color1);
  line-height: 24px;
  display: block;
}

.jp-Notebook-ExecutionIndicator-progress-bar {
  display: flex;
  justify-content: center;
  height: 100%;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*
 * Execution indicator
 */
.jp-tocItem-content::after {
  content: '';

  /* Must be identical to form a circle */
  width: 12px;
  height: 12px;
  background: none;
  border: none;
  position: absolute;
  right: 0;
}

.jp-tocItem-content[data-running='0']::after {
  border-radius: 50%;
  border: var(--jp-border-width) solid var(--jp-inverse-layout-color3);
  background: none;
}

.jp-tocItem-content[data-running='1']::after {
  border-radius: 50%;
  border: var(--jp-border-width) solid var(--jp-inverse-layout-color3);
  background-color: var(--jp-inverse-layout-color3);
}

.jp-tocItem-content[data-running='0'],
.jp-tocItem-content[data-running='1'] {
  margin-right: 12px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-Notebook-footer {
  height: 27px;
  margin-left: calc(
    var(--jp-cell-prompt-width) + var(--jp-cell-collapser-width) +
      var(--jp-cell-padding)
  );
  width: calc(
    100% -
      (
        var(--jp-cell-prompt-width) + var(--jp-cell-collapser-width) +
          var(--jp-cell-padding) + var(--jp-cell-padding)
      )
  );
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  color: var(--jp-ui-font-color3);
  margin-top: 6px;
  background: none;
  cursor: pointer;
}

.jp-Notebook-footer:focus {
  border-color: var(--jp-cell-editor-active-border-color);
}

/* For devices that support hovering, hide footer until hover */
@media (hover: hover) {
  .jp-Notebook-footer {
    opacity: 0;
  }

  .jp-Notebook-footer:focus,
  .jp-Notebook-footer:hover {
    opacity: 1;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Imports
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-side-by-side-output-size: 1fr;
  --jp-side-by-side-resized-cell: var(--jp-side-by-side-output-size);
  --jp-private-notebook-dragImage-width: 304px;
  --jp-private-notebook-dragImage-height: 36px;
  --jp-private-notebook-selected-color: var(--md-blue-400);
  --jp-private-notebook-active-color: var(--md-green-400);
}

/*-----------------------------------------------------------------------------
| Notebook
|----------------------------------------------------------------------------*/

/* stylelint-disable selector-max-class */

.jp-NotebookPanel {
  display: block;
  height: 100%;
}

.jp-NotebookPanel.jp-Document {
  min-width: 240px;
  min-height: 120px;
}

.jp-Notebook {
  padding: var(--jp-notebook-padding);
  outline: none;
  overflow: auto;
  background: var(--jp-layout-color0);
}

.jp-Notebook.jp-mod-scrollPastEnd::after {
  display: block;
  content: '';
  min-height: var(--jp-notebook-scroll-padding);
}

.jp-MainAreaWidget-ContainStrict .jp-Notebook * {
  contain: strict;
}

.jp-Notebook .jp-Cell {
  overflow: visible;
}

.jp-Notebook .jp-Cell .jp-InputPrompt {
  cursor: move;
}

/*-----------------------------------------------------------------------------
| Notebook state related styling
|
| The notebook and cells each have states, here are the possibilities:
|
| - Notebook
|   - Command
|   - Edit
| - Cell
|   - None
|   - Active (only one can be active)
|   - Selected (the cells actions are applied to)
|   - Multiselected (when multiple selected, the cursor)
|   - No outputs
|----------------------------------------------------------------------------*/

/* Command or edit modes */

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-InputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-OutputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

/* cell is active */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser {
  background: var(--jp-brand-color1);
}

/* cell is dirty */
.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt {
  color: var(--jp-warn-color1);
}

.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt::before {
  color: var(--jp-warn-color1);
  content: '•';
}

.jp-Notebook .jp-Cell.jp-mod-active.jp-mod-dirty .jp-Collapser {
  background: var(--jp-warn-color1);
}

/* collapser is hovered */
.jp-Notebook .jp-Cell .jp-Collapser:hover {
  box-shadow: var(--jp-elevation-z2);
  background: var(--jp-brand-color1);
  opacity: var(--jp-cell-collapser-not-active-hover-opacity);
}

/* cell is active and collapser is hovered */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser:hover {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/* Command mode */

.jp-Notebook.jp-mod-commandMode .jp-Cell.jp-mod-selected {
  background: var(--jp-notebook-multiselected-color);
}

.jp-Notebook.jp-mod-commandMode
  .jp-Cell.jp-mod-active.jp-mod-selected:not(.jp-mod-multiSelected) {
  background: transparent;
}

/* Edit mode */

.jp-Notebook.jp-mod-editMode .jp-Cell.jp-mod-active .jp-InputArea-editor {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-cell-editor-active-background);
}

/*-----------------------------------------------------------------------------
| Notebook drag and drop
|----------------------------------------------------------------------------*/

.jp-Notebook-cell.jp-mod-dropSource {
  opacity: 0.5;
}

.jp-Notebook-cell.jp-mod-dropTarget,
.jp-Notebook.jp-mod-commandMode
  .jp-Notebook-cell.jp-mod-active.jp-mod-selected.jp-mod-dropTarget {
  border-top-color: var(--jp-private-notebook-selected-color);
  border-top-style: solid;
  border-top-width: 2px;
}

.jp-dragImage {
  display: block;
  flex-direction: row;
  width: var(--jp-private-notebook-dragImage-width);
  height: var(--jp-private-notebook-dragImage-height);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
  overflow: visible;
}

.jp-dragImage-singlePrompt {
  box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.12);
}

.jp-dragImage .jp-dragImage-content {
  flex: 1 1 auto;
  z-index: 2;
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  line-height: var(--jp-code-line-height);
  padding: var(--jp-code-padding);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background-color);
  color: var(--jp-content-font-color3);
  text-align: left;
  margin: 4px 4px 4px 0;
}

.jp-dragImage .jp-dragImage-prompt {
  flex: 0 0 auto;
  min-width: 36px;
  color: var(--jp-cell-inprompt-font-color);
  padding: var(--jp-code-padding);
  padding-left: 12px;
  font-family: var(--jp-cell-prompt-font-family);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: 1.9;
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
}

.jp-dragImage-multipleBack {
  z-index: -1;
  position: absolute;
  height: 32px;
  width: 300px;
  top: 8px;
  left: 8px;
  background: var(--jp-layout-color2);
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.12);
}

/*-----------------------------------------------------------------------------
| Cell toolbar
|----------------------------------------------------------------------------*/

.jp-NotebookTools {
  display: block;
  min-width: var(--jp-sidebar-min-width);
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);

  /* This is needed so that all font sizing of children done in ems is
    * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  overflow: auto;
}

.jp-ActiveCellTool {
  padding: 12px 0;
  display: flex;
}

.jp-ActiveCellTool-Content {
  flex: 1 1 auto;
}

.jp-ActiveCellTool .jp-ActiveCellTool-CellContent {
  background: var(--jp-cell-editor-background);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0;
  min-height: 29px;
}

.jp-ActiveCellTool .jp-InputPrompt {
  min-width: calc(var(--jp-cell-prompt-width) * 0.75);
}

.jp-ActiveCellTool-CellContent > pre {
  padding: 5px 4px;
  margin: 0;
  white-space: normal;
}

.jp-MetadataEditorTool {
  flex-direction: column;
  padding: 12px 0;
}

.jp-RankedPanel > :not(:first-child) {
  margin-top: 12px;
}

.jp-KeySelector select.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: var(--jp-border-width) solid var(--jp-border-color1);
}

.jp-KeySelector label,
.jp-MetadataEditorTool label,
.jp-NumberSetter label {
  line-height: 1.4;
}

.jp-NotebookTools .jp-select-wrapper {
  margin-top: 4px;
  margin-bottom: 0;
}

.jp-NumberSetter input {
  width: 100%;
  margin-top: 4px;
}

.jp-NotebookTools .jp-Collapse {
  margin-top: 16px;
}

/*-----------------------------------------------------------------------------
| Presentation Mode (.jp-mod-presentationMode)
|----------------------------------------------------------------------------*/

.jp-mod-presentationMode .jp-Notebook {
  --jp-content-font-size1: var(--jp-content-presentation-font-size1);
  --jp-code-font-size: var(--jp-code-presentation-font-size);
}

.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-InputPrompt,
.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-OutputPrompt {
  flex: 0 0 110px;
}

/*-----------------------------------------------------------------------------
| Side-by-side Mode (.jp-mod-sideBySide)
|----------------------------------------------------------------------------*/
.jp-mod-sideBySide.jp-Notebook .jp-Notebook-cell {
  margin-top: 3em;
  margin-bottom: 3em;
  margin-left: 5%;
  margin-right: 5%;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell {
  display: grid;
  grid-template-columns: minmax(0, 1fr) min-content minmax(
      0,
      var(--jp-side-by-side-output-size)
    );
  grid-template-rows: auto minmax(0, 1fr) auto;
  grid-template-areas:
    'header header header'
    'input handle output'
    'footer footer footer';
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell.jp-mod-resizedCell {
  grid-template-columns: minmax(0, 1fr) min-content minmax(
      0,
      var(--jp-side-by-side-resized-cell)
    );
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellHeader {
  grid-area: header;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-Cell-inputWrapper {
  grid-area: input;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-Cell-outputWrapper {
  /* overwrite the default margin (no vertical separation needed in side by side move */
  margin-top: 0;
  grid-area: output;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellFooter {
  grid-area: footer;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellResizeHandle {
  grid-area: handle;
  user-select: none;
  display: block;
  height: 100%;
  cursor: ew-resize;
  padding: 0 var(--jp-cell-padding);
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellResizeHandle::after {
  content: '';
  display: block;
  background: var(--jp-border-color2);
  height: 100%;
  width: 5px;
}

.jp-mod-sideBySide.jp-Notebook
  .jp-CodeCell.jp-mod-resizedCell
  .jp-CellResizeHandle::after {
  background: var(--jp-border-color0);
}

.jp-CellResizeHandle {
  display: none;
}

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Cell-Placeholder {
  padding-left: 55px;
}

.jp-Cell-Placeholder-wrapper {
  background: #fff;
  border: 1px solid;
  border-color: #e5e6e9 #dfe0e4 #d0d1d5;
  border-radius: 4px;
  -webkit-border-radius: 4px;
  margin: 10px 15px;
}

.jp-Cell-Placeholder-wrapper-inner {
  padding: 15px;
  position: relative;
}

.jp-Cell-Placeholder-wrapper-body {
  background-repeat: repeat;
  background-size: 50% auto;
}

.jp-Cell-Placeholder-wrapper-body div {
  background: #f6f7f8;
  background-image: -webkit-linear-gradient(
    left,
    #f6f7f8 0%,
    #edeef1 20%,
    #f6f7f8 40%,
    #f6f7f8 100%
  );
  background-repeat: no-repeat;
  background-size: 800px 104px;
  height: 104px;
  position: absolute;
  right: 15px;
  left: 15px;
  top: 15px;
}

div.jp-Cell-Placeholder-h1 {
  top: 20px;
  height: 20px;
  left: 15px;
  width: 150px;
}

div.jp-Cell-Placeholder-h2 {
  left: 15px;
  top: 50px;
  height: 10px;
  width: 100px;
}

div.jp-Cell-Placeholder-content-1,
div.jp-Cell-Placeholder-content-2,
div.jp-Cell-Placeholder-content-3 {
  left: 15px;
  right: 15px;
  height: 10px;
}

div.jp-Cell-Placeholder-content-1 {
  top: 100px;
}

div.jp-Cell-Placeholder-content-2 {
  top: 120px;
}

div.jp-Cell-Placeholder-content-3 {
  top: 140px;
}

</style>
<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
The following CSS variables define the main, public API for styling JupyterLab.
These variables should be used by all plugins wherever possible. In other
words, plugins should not define custom colors, sizes, etc unless absolutely
necessary. This enables users to change the visual theme of JupyterLab
by changing these variables.

Many variables appear in an ordered sequence (0,1,2,3). These sequences
are designed to work well together, so for example, `--jp-border-color1` should
be used with `--jp-layout-color1`. The numbers have the following meanings:

* 0: super-primary, reserved for special emphasis
* 1: primary, most important under normal situations
* 2: secondary, next most important under normal situations
* 3: tertiary, next most important under normal situations

Throughout JupyterLab, we are mostly following principles from Google's
Material Design when selecting colors. We are not, however, following
all of MD as it is not optimized for dense, information rich UIs.
*/

:root {
  /* Elevation
   *
   * We style box-shadows using Material Design's idea of elevation. These particular numbers are taken from here:
   *
   * https://github.com/material-components/material-components-web
   * https://material-components-web.appspot.com/elevation.html
   */

  --jp-shadow-base-lightness: 0;
  --jp-shadow-umbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.2
  );
  --jp-shadow-penumbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.14
  );
  --jp-shadow-ambient-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.12
  );
  --jp-elevation-z0: none;
  --jp-elevation-z1: 0 2px 1px -1px var(--jp-shadow-umbra-color),
    0 1px 1px 0 var(--jp-shadow-penumbra-color),
    0 1px 3px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z2: 0 3px 1px -2px var(--jp-shadow-umbra-color),
    0 2px 2px 0 var(--jp-shadow-penumbra-color),
    0 1px 5px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z4: 0 2px 4px -1px var(--jp-shadow-umbra-color),
    0 4px 5px 0 var(--jp-shadow-penumbra-color),
    0 1px 10px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z6: 0 3px 5px -1px var(--jp-shadow-umbra-color),
    0 6px 10px 0 var(--jp-shadow-penumbra-color),
    0 1px 18px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z8: 0 5px 5px -3px var(--jp-shadow-umbra-color),
    0 8px 10px 1px var(--jp-shadow-penumbra-color),
    0 3px 14px 2px var(--jp-shadow-ambient-color);
  --jp-elevation-z12: 0 7px 8px -4px var(--jp-shadow-umbra-color),
    0 12px 17px 2px var(--jp-shadow-penumbra-color),
    0 5px 22px 4px var(--jp-shadow-ambient-color);
  --jp-elevation-z16: 0 8px 10px -5px var(--jp-shadow-umbra-color),
    0 16px 24px 2px var(--jp-shadow-penumbra-color),
    0 6px 30px 5px var(--jp-shadow-ambient-color);
  --jp-elevation-z20: 0 10px 13px -6px var(--jp-shadow-umbra-color),
    0 20px 31px 3px var(--jp-shadow-penumbra-color),
    0 8px 38px 7px var(--jp-shadow-ambient-color);
  --jp-elevation-z24: 0 11px 15px -7px var(--jp-shadow-umbra-color),
    0 24px 38px 3px var(--jp-shadow-penumbra-color),
    0 9px 46px 8px var(--jp-shadow-ambient-color);

  /* Borders
   *
   * The following variables, specify the visual styling of borders in JupyterLab.
   */

  --jp-border-width: 1px;
  --jp-border-color0: var(--md-grey-400);
  --jp-border-color1: var(--md-grey-400);
  --jp-border-color2: var(--md-grey-300);
  --jp-border-color3: var(--md-grey-200);
  --jp-inverse-border-color: var(--md-grey-600);
  --jp-border-radius: 2px;

  /* UI Fonts
   *
   * The UI font CSS variables are used for the typography all of the JupyterLab
   * user interface elements that are not directly user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-ui-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-ui-font-scale-factor: 1.2;
  --jp-ui-font-size0: 0.83333em;
  --jp-ui-font-size1: 13px; /* Base font size */
  --jp-ui-font-size2: 1.2em;
  --jp-ui-font-size3: 1.44em;
  --jp-ui-font-family: system-ui, -apple-system, blinkmacsystemfont, 'Segoe UI',
    helvetica, arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji',
    'Segoe UI Symbol';

  /*
   * Use these font colors against the corresponding main layout colors.
   * In a light theme, these go from dark to light.
   */

  /* Defaults use Material Design specification */
  --jp-ui-font-color0: rgba(0, 0, 0, 1);
  --jp-ui-font-color1: rgba(0, 0, 0, 0.87);
  --jp-ui-font-color2: rgba(0, 0, 0, 0.54);
  --jp-ui-font-color3: rgba(0, 0, 0, 0.38);

  /*
   * Use these against the brand/accent/warn/error colors.
   * These will typically go from light to darker, in both a dark and light theme.
   */

  --jp-ui-inverse-font-color0: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color1: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color2: rgba(255, 255, 255, 0.7);
  --jp-ui-inverse-font-color3: rgba(255, 255, 255, 0.5);

  /* Content Fonts
   *
   * Content font variables are used for typography of user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-content-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-content-line-height: 1.6;
  --jp-content-font-scale-factor: 1.2;
  --jp-content-font-size0: 0.83333em;
  --jp-content-font-size1: 14px; /* Base font size */
  --jp-content-font-size2: 1.2em;
  --jp-content-font-size3: 1.44em;
  --jp-content-font-size4: 1.728em;
  --jp-content-font-size5: 2.0736em;

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-content-presentation-font-size1: 17px;
  --jp-content-heading-line-height: 1;
  --jp-content-heading-margin-top: 1.2em;
  --jp-content-heading-margin-bottom: 0.8em;
  --jp-content-heading-font-weight: 500;

  /* Defaults use Material Design specification */
  --jp-content-font-color0: rgba(0, 0, 0, 1);
  --jp-content-font-color1: rgba(0, 0, 0, 0.87);
  --jp-content-font-color2: rgba(0, 0, 0, 0.54);
  --jp-content-font-color3: rgba(0, 0, 0, 0.38);
  --jp-content-link-color: var(--md-blue-900);
  --jp-content-font-family: system-ui, -apple-system, blinkmacsystemfont,
    'Segoe UI', helvetica, arial, sans-serif, 'Apple Color Emoji',
    'Segoe UI Emoji', 'Segoe UI Symbol';

  /*
   * Code Fonts
   *
   * Code font variables are used for typography of code and other monospaces content.
   */

  --jp-code-font-size: 13px;
  --jp-code-line-height: 1.3077; /* 17px for 13px base */
  --jp-code-padding: 5px; /* 5px for 13px base, codemirror highlighting needs integer px value */
  --jp-code-font-family-default: menlo, consolas, 'DejaVu Sans Mono', monospace;
  --jp-code-font-family: var(--jp-code-font-family-default);

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-code-presentation-font-size: 16px;

  /* may need to tweak cursor width if you change font size */
  --jp-code-cursor-width0: 1.4px;
  --jp-code-cursor-width1: 2px;
  --jp-code-cursor-width2: 4px;

  /* Layout
   *
   * The following are the main layout colors use in JupyterLab. In a light
   * theme these would go from light to dark.
   */

  --jp-layout-color0: white;
  --jp-layout-color1: white;
  --jp-layout-color2: var(--md-grey-200);
  --jp-layout-color3: var(--md-grey-400);
  --jp-layout-color4: var(--md-grey-600);

  /* Inverse Layout
   *
   * The following are the inverse layout colors use in JupyterLab. In a light
   * theme these would go from dark to light.
   */

  --jp-inverse-layout-color0: #111;
  --jp-inverse-layout-color1: var(--md-grey-900);
  --jp-inverse-layout-color2: var(--md-grey-800);
  --jp-inverse-layout-color3: var(--md-grey-700);
  --jp-inverse-layout-color4: var(--md-grey-600);

  /* Brand/accent */

  --jp-brand-color0: var(--md-blue-900);
  --jp-brand-color1: var(--md-blue-700);
  --jp-brand-color2: var(--md-blue-300);
  --jp-brand-color3: var(--md-blue-100);
  --jp-brand-color4: var(--md-blue-50);
  --jp-accent-color0: var(--md-green-900);
  --jp-accent-color1: var(--md-green-700);
  --jp-accent-color2: var(--md-green-300);
  --jp-accent-color3: var(--md-green-100);

  /* State colors (warn, error, success, info) */

  --jp-warn-color0: var(--md-orange-900);
  --jp-warn-color1: var(--md-orange-700);
  --jp-warn-color2: var(--md-orange-300);
  --jp-warn-color3: var(--md-orange-100);
  --jp-error-color0: var(--md-red-900);
  --jp-error-color1: var(--md-red-700);
  --jp-error-color2: var(--md-red-300);
  --jp-error-color3: var(--md-red-100);
  --jp-success-color0: var(--md-green-900);
  --jp-success-color1: var(--md-green-700);
  --jp-success-color2: var(--md-green-300);
  --jp-success-color3: var(--md-green-100);
  --jp-info-color0: var(--md-cyan-900);
  --jp-info-color1: var(--md-cyan-700);
  --jp-info-color2: var(--md-cyan-300);
  --jp-info-color3: var(--md-cyan-100);

  /* Cell specific styles */

  --jp-cell-padding: 5px;
  --jp-cell-collapser-width: 8px;
  --jp-cell-collapser-min-height: 20px;
  --jp-cell-collapser-not-active-hover-opacity: 0.6;
  --jp-cell-editor-background: var(--md-grey-100);
  --jp-cell-editor-border-color: var(--md-grey-300);
  --jp-cell-editor-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-cell-editor-active-background: var(--jp-layout-color0);
  --jp-cell-editor-active-border-color: var(--jp-brand-color1);
  --jp-cell-prompt-width: 64px;
  --jp-cell-prompt-font-family: var(--jp-code-font-family-default);
  --jp-cell-prompt-letter-spacing: 0;
  --jp-cell-prompt-opacity: 1;
  --jp-cell-prompt-not-active-opacity: 0.5;
  --jp-cell-prompt-not-active-font-color: var(--md-grey-700);

  /* A custom blend of MD grey and blue 600
   * See https://meyerweb.com/eric/tools/color-blend/#546E7A:1E88E5:5:hex */
  --jp-cell-inprompt-font-color: #307fc1;

  /* A custom blend of MD grey and orange 600
   * https://meyerweb.com/eric/tools/color-blend/#546E7A:F4511E:5:hex */
  --jp-cell-outprompt-font-color: #bf5b3d;

  /* Notebook specific styles */

  --jp-notebook-padding: 10px;
  --jp-notebook-select-background: var(--jp-layout-color1);
  --jp-notebook-multiselected-color: var(--md-blue-50);

  /* The scroll padding is calculated to fill enough space at the bottom of the
  notebook to show one single-line cell (with appropriate padding) at the top
  when the notebook is scrolled all the way to the bottom. We also subtract one
  pixel so that no scrollbar appears if we have just one single-line cell in the
  notebook. This padding is to enable a 'scroll past end' feature in a notebook.
  */
  --jp-notebook-scroll-padding: calc(
    100% - var(--jp-code-font-size) * var(--jp-code-line-height) -
      var(--jp-code-padding) - var(--jp-cell-padding) - 1px
  );

  /* Rendermime styles */

  --jp-rendermime-error-background: #fdd;
  --jp-rendermime-table-row-background: var(--md-grey-100);
  --jp-rendermime-table-row-hover-background: var(--md-light-blue-50);

  /* Dialog specific styles */

  --jp-dialog-background: rgba(0, 0, 0, 0.25);

  /* Console specific styles */

  --jp-console-padding: 10px;

  /* Toolbar specific styles */

  --jp-toolbar-border-color: var(--jp-border-color1);
  --jp-toolbar-micro-height: 8px;
  --jp-toolbar-background: var(--jp-layout-color1);
  --jp-toolbar-box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.24);
  --jp-toolbar-header-margin: 4px 4px 0 4px;
  --jp-toolbar-active-background: var(--md-grey-300);

  /* Statusbar specific styles */

  --jp-statusbar-height: 24px;

  /* Input field styles */

  --jp-input-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-input-active-background: var(--jp-layout-color1);
  --jp-input-hover-background: var(--jp-layout-color1);
  --jp-input-background: var(--md-grey-100);
  --jp-input-border-color: var(--jp-inverse-border-color);
  --jp-input-active-border-color: var(--jp-brand-color1);
  --jp-input-active-box-shadow-color: rgba(19, 124, 189, 0.3);

  /* General editor styles */

  --jp-editor-selected-background: #d9d9d9;
  --jp-editor-selected-focused-background: #d7d4f0;
  --jp-editor-cursor-color: var(--jp-ui-font-color0);

  /* Code mirror specific styles */

  --jp-mirror-editor-keyword-color: #008000;
  --jp-mirror-editor-atom-color: #88f;
  --jp-mirror-editor-number-color: #080;
  --jp-mirror-editor-def-color: #00f;
  --jp-mirror-editor-variable-color: var(--md-grey-900);
  --jp-mirror-editor-variable-2-color: rgb(0, 54, 109);
  --jp-mirror-editor-variable-3-color: #085;
  --jp-mirror-editor-punctuation-color: #05a;
  --jp-mirror-editor-property-color: #05a;
  --jp-mirror-editor-operator-color: #a2f;
  --jp-mirror-editor-comment-color: #408080;
  --jp-mirror-editor-string-color: #ba2121;
  --jp-mirror-editor-string-2-color: #708;
  --jp-mirror-editor-meta-color: #a2f;
  --jp-mirror-editor-qualifier-color: #555;
  --jp-mirror-editor-builtin-color: #008000;
  --jp-mirror-editor-bracket-color: #997;
  --jp-mirror-editor-tag-color: #170;
  --jp-mirror-editor-attribute-color: #00c;
  --jp-mirror-editor-header-color: blue;
  --jp-mirror-editor-quote-color: #090;
  --jp-mirror-editor-link-color: #00c;
  --jp-mirror-editor-error-color: #f00;
  --jp-mirror-editor-hr-color: #999;

  /*
    RTC user specific colors.
    These colors are used for the cursor, username in the editor,
    and the icon of the user.
  */

  --jp-collaborator-color1: #ffad8e;
  --jp-collaborator-color2: #dac83d;
  --jp-collaborator-color3: #72dd76;
  --jp-collaborator-color4: #00e4d0;
  --jp-collaborator-color5: #45d4ff;
  --jp-collaborator-color6: #e2b1ff;
  --jp-collaborator-color7: #ff9de6;

  /* Vega extension styles */

  --jp-vega-background: white;

  /* Sidebar-related styles */

  --jp-sidebar-min-width: 250px;

  /* Search-related styles */

  --jp-search-toggle-off-opacity: 0.5;
  --jp-search-toggle-hover-opacity: 0.8;
  --jp-search-toggle-on-opacity: 1;
  --jp-search-selected-match-background-color: rgb(245, 200, 0);
  --jp-search-selected-match-color: black;
  --jp-search-unselected-match-background-color: var(
    --jp-inverse-layout-color0
  );
  --jp-search-unselected-match-color: var(--jp-ui-inverse-font-color0);

  /* Icon colors that work well with light or dark backgrounds */
  --jp-icon-contrast-color0: var(--md-purple-600);
  --jp-icon-contrast-color1: var(--md-green-600);
  --jp-icon-contrast-color2: var(--md-pink-600);
  --jp-icon-contrast-color3: var(--md-blue-600);

  /* Button colors */
  --jp-accept-color-normal: var(--md-blue-700);
  --jp-accept-color-hover: var(--md-blue-800);
  --jp-accept-color-active: var(--md-blue-900);
  --jp-warn-color-normal: var(--md-red-700);
  --jp-warn-color-hover: var(--md-red-800);
  --jp-warn-color-active: var(--md-red-900);
  --jp-reject-color-normal: var(--md-grey-600);
  --jp-reject-color-hover: var(--md-grey-700);
  --jp-reject-color-active: var(--md-grey-800);

  /* File or activity icons and switch semantic variables */
  --jp-jupyter-icon-color: #f37626;
  --jp-notebook-icon-color: #f37626;
  --jp-json-icon-color: var(--md-orange-700);
  --jp-console-icon-background-color: var(--md-blue-700);
  --jp-console-icon-color: white;
  --jp-terminal-icon-background-color: var(--md-grey-800);
  --jp-terminal-icon-color: var(--md-grey-200);
  --jp-text-editor-icon-color: var(--md-grey-700);
  --jp-inspector-icon-color: var(--md-grey-700);
  --jp-switch-color: var(--md-grey-400);
  --jp-switch-true-position-color: var(--md-orange-900);
}
</style>
<style type="text/css">
/* Force rendering true colors when outputing to pdf */
* {
  -webkit-print-color-adjust: exact;
}

/* Misc */
a.anchor-link {
  display: none;
}

/* Input area styling */
.jp-InputArea {
  overflow: hidden;
}

.jp-InputArea-editor {
  overflow: hidden;
}

.cm-editor.cm-s-jupyter .highlight pre {
/* weird, but --jp-code-padding defined to be 5px but 4px horizontal padding is hardcoded for pre.cm-line */
  padding: var(--jp-code-padding) 4px;
  margin: 0;

  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
  color: inherit;

}

.jp-OutputArea-output pre {
  line-height: inherit;
  font-family: inherit;
}

.jp-RenderedText pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
}

/* Hiding the collapser by default */
.jp-Collapser {
  display: none;
}

@page {
    margin: 0.5in; /* Margin for each printed piece of paper */
}

@media print {
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }
}
</style>
<!-- Load mathjax -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML-full,Safe"> </script>
<!-- MathJax configuration -->
<script type="text/x-mathjax-config">
    init_mathjax = function() {
        if (window.MathJax) {
        // MathJax loaded
            MathJax.Hub.Config({
                TeX: {
                    equationNumbers: {
                    autoNumber: "AMS",
                    useLabelIds: true
                    }
                },
                tex2jax: {
                    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
                    displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
                    processEscapes: true,
                    processEnvironments: true
                },
                displayAlign: 'center',
                CommonHTML: {
                    linebreaks: {
                    automatic: true
                    }
                }
            });

            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    }
    init_mathjax();
    </script>
<!-- End of mathjax configuration --><script type="module">
  document.addEventListener("DOMContentLoaded", async () => {
    const diagrams = document.querySelectorAll(".jp-Mermaid > pre.mermaid");
    // do not load mermaidjs if not needed
    if (!diagrams.length) {
      return;
    }
    const mermaid = (await import("https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.6.0/mermaid.esm.min.mjs")).default;
    const parser = new DOMParser();

    mermaid.initialize({
      maxTextSize: 100000,
      startOnLoad: false,
      fontFamily: window
        .getComputedStyle(document.body)
        .getPropertyValue("--jp-ui-font-family"),
      theme: document.querySelector("body[data-jp-theme-light='true']")
        ? "default"
        : "dark",
    });

    let _nextMermaidId = 0;

    function makeMermaidImage(svg) {
      const img = document.createElement("img");
      const doc = parser.parseFromString(svg, "image/svg+xml");
      const svgEl = doc.querySelector("svg");
      const { maxWidth } = svgEl?.style || {};
      const firstTitle = doc.querySelector("title");
      const firstDesc = doc.querySelector("desc");

      img.setAttribute("src", `data:image/svg+xml,${encodeURIComponent(svg)}`);
      if (maxWidth) {
        img.width = parseInt(maxWidth);
      }
      if (firstTitle) {
        img.setAttribute("alt", firstTitle.textContent);
      }
      if (firstDesc) {
        const caption = document.createElement("figcaption");
        caption.className = "sr-only";
        caption.textContent = firstDesc.textContent;
        return [img, caption];
      }
      return [img];
    }

    async function makeMermaidError(text) {
      let errorMessage = "";
      try {
        await mermaid.parse(text);
      } catch (err) {
        errorMessage = `${err}`;
      }

      const result = document.createElement("details");
      result.className = 'jp-RenderedMermaid-Details';
      const summary = document.createElement("summary");
      summary.className = 'jp-RenderedMermaid-Summary';
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.innerText = text;
      pre.appendChild(code);
      summary.appendChild(pre);
      result.appendChild(summary);

      const warning = document.createElement("pre");
      warning.innerText = errorMessage;
      result.appendChild(warning);
      return [result];
    }

    async function renderOneMarmaid(src) {
      const id = `jp-mermaid-${_nextMermaidId++}`;
      const parent = src.parentNode;
      let raw = src.textContent.trim();
      const el = document.createElement("div");
      el.style.visibility = "hidden";
      document.body.appendChild(el);
      let results = null;
      let output = null;
      try {
        const { svg } = await mermaid.render(id, raw, el);
        results = makeMermaidImage(svg);
        output = document.createElement("figure");
        results.map(output.appendChild, output);
      } catch (err) {
        parent.classList.add("jp-mod-warning");
        results = await makeMermaidError(raw);
        output = results[0];
      } finally {
        el.remove();
      }
      parent.classList.add("jp-RenderedMermaid");
      parent.appendChild(output);
    }

    void Promise.all([...diagrams].map(renderOneMarmaid));
  });
</script>
<style>
  .jp-Mermaid:not(.jp-RenderedMermaid) {
    display: none;
  }

  .jp-RenderedMermaid {
    overflow: auto;
    display: flex;
  }

  .jp-RenderedMermaid.jp-mod-warning {
    width: auto;
    padding: 0.5em;
    margin-top: 0.5em;
    border: var(--jp-border-width) solid var(--jp-warn-color2);
    border-radius: var(--jp-border-radius);
    color: var(--jp-ui-font-color1);
    font-size: var(--jp-ui-font-size1);
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  .jp-RenderedMermaid figure {
    margin: 0;
    overflow: auto;
    max-width: 100%;
  }

  .jp-RenderedMermaid img {
    max-width: 100%;
  }

  .jp-RenderedMermaid-Details > pre {
    margin-top: 1em;
  }

  .jp-RenderedMermaid-Summary {
    color: var(--jp-warn-color2);
  }

  .jp-RenderedMermaid:not(.jp-mod-warning) pre {
    display: none;
  }

  .jp-RenderedMermaid-Summary > pre {
    display: inline-block;
    white-space: normal;
  }
</style>
<!-- End of mermaid configuration --></head>
<body class="jp-Notebook" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light">
<main>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h2 id="Movie-Data-Analysis">Movie Data Analysis<a class="anchor-link" href="#Movie-Data-Analysis">¶</a></h2>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h4 id="Your-goal-is-to-Analyze-the-data-and-find-the-Highest-correlated-features-in-the-data-using-Python-and-listed-libraries">Your goal is to Analyze the data and find the Highest correlated features in the data using Python and listed libraries<a class="anchor-link" href="#Your-goal-is-to-Analyze-the-data-and-find-the-Highest-correlated-features-in-the-data-using-Python-and-listed-libraries">¶</a></h4><h4 id="Libraries-used-are:">Libraries used are:<a class="anchor-link" href="#Libraries-used-are:">¶</a></h4><h4 id="--Pandas-%E2%86%92-for-holding-data-and-structure-the-data">- Pandas → for holding data and structure the data<a class="anchor-link" href="#--Pandas-%E2%86%92-for-holding-data-and-structure-the-data">¶</a></h4><h4 id="--Seaborn-&amp;-Matplotlib-%E2%86%92-for-visualizing-data-features">- Seaborn &amp; Matplotlib → for visualizing data features<a class="anchor-link" href="#--Seaborn-&amp;-Matplotlib-%E2%86%92-for-visualizing-data-features">¶</a></h4><h4 id="--Numpay-%E2%86%92-for-data-manipulation-and-data-structure-needs">- Numpay → for data manipulation and data structure needs<a class="anchor-link" href="#--Numpay-%E2%86%92-for-data-manipulation-and-data-structure-needs">¶</a></h4>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">matplotlib</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="n">plt</span><span class="o">.</span><span class="n">style</span><span class="o">.</span><span class="n">use</span><span class="p">(</span><span class="s1">'ggplot'</span><span class="p">)</span>

<span class="o">%</span><span class="k">matplotlib</span> inline
<span class="n">matplotlib</span><span class="o">.</span><span class="n">rcParams</span><span class="p">[</span><span class="s1">'figure.figsize'</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span><span class="mi">12</span><span class="p">,</span><span class="mi">8</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h4 id="Read-Data-from-File">Read Data from File<a class="anchor-link" href="#Read-Data-from-File">¶</a></h4>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1">#read data</span>
<span class="n">df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">'./movies.csv'</span><span class="p">)</span>
<span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[ ]:</div>
<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html" tabindex="0">
<div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>name</th>
<th>rating</th>
<th>genre</th>
<th>year</th>
<th>released</th>
<th>score</th>
<th>votes</th>
<th>director</th>
<th>writer</th>
<th>star</th>
<th>country</th>
<th>budget</th>
<th>gross</th>
<th>company</th>
<th>runtime</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>The Shining</td>
<td>R</td>
<td>Drama</td>
<td>1980</td>
<td>June 13, 1980 (United States)</td>
<td>8.4</td>
<td>927000.0</td>
<td>Stanley Kubrick</td>
<td>Stephen King</td>
<td>Jack Nicholson</td>
<td>United Kingdom</td>
<td>19000000.0</td>
<td>46998772.0</td>
<td>Warner Bros.</td>
<td>146.0</td>
</tr>
<tr>
<th>1</th>
<td>The Blue Lagoon</td>
<td>R</td>
<td>Adventure</td>
<td>1980</td>
<td>July 2, 1980 (United States)</td>
<td>5.8</td>
<td>65000.0</td>
<td>Randal Kleiser</td>
<td>Henry De Vere Stacpoole</td>
<td>Brooke Shields</td>
<td>United States</td>
<td>4500000.0</td>
<td>58853106.0</td>
<td>Columbia Pictures</td>
<td>104.0</td>
</tr>
<tr>
<th>2</th>
<td>Star Wars: Episode V - The Empire Strikes Back</td>
<td>PG</td>
<td>Action</td>
<td>1980</td>
<td>June 20, 1980 (United States)</td>
<td>8.7</td>
<td>1200000.0</td>
<td>Irvin Kershner</td>
<td>Leigh Brackett</td>
<td>Mark Hamill</td>
<td>United States</td>
<td>18000000.0</td>
<td>538375067.0</td>
<td>Lucasfilm</td>
<td>124.0</td>
</tr>
<tr>
<th>3</th>
<td>Airplane!</td>
<td>PG</td>
<td>Comedy</td>
<td>1980</td>
<td>July 2, 1980 (United States)</td>
<td>7.7</td>
<td>221000.0</td>
<td>Jim Abrahams</td>
<td>Jim Abrahams</td>
<td>Robert Hays</td>
<td>United States</td>
<td>3500000.0</td>
<td>83453539.0</td>
<td>Paramount Pictures</td>
<td>88.0</td>
</tr>
<tr>
<th>4</th>
<td>Caddyshack</td>
<td>R</td>
<td>Comedy</td>
<td>1980</td>
<td>July 25, 1980 (United States)</td>
<td>7.3</td>
<td>108000.0</td>
<td>Harold Ramis</td>
<td>Brian Doyle-Murray</td>
<td>Chevy Chase</td>
<td>United States</td>
<td>6000000.0</td>
<td>39846344.0</td>
<td>Orion Pictures</td>
<td>98.0</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h2 id="Data-Cleaning">Data Cleaning<a class="anchor-link" href="#Data-Cleaning">¶</a></h2><h4 id="Finding-missing-value">Finding missing value<a class="anchor-link" href="#Finding-missing-value">¶</a></h4>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># data cleaning</span>
<span class="c1"># - find missing values </span>
<span class="k">for</span> <span class="n">col</span> <span class="ow">in</span> <span class="n">df</span><span class="o">.</span><span class="n">columns</span><span class="p">:</span>
    <span class="n">pct_missing</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="n">col</span><span class="p">]</span><span class="o">.</span><span class="n">isnull</span><span class="p">())</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">'</span><span class="si">{</span><span class="n">col</span><span class="si">}</span><span class="s1"> - </span><span class="si">{</span><span class="n">pct_missing</span><span class="si">}</span><span class="s1">%'</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>name - 0.0%
rating - 0.010041731872717789%
genre - 0.0%
year - 0.0%
released - 0.0002608242044861763%
score - 0.0003912363067292645%
votes - 0.0003912363067292645%
director - 0.0%
writer - 0.0003912363067292645%
star - 0.00013041210224308815%
country - 0.0003912363067292645%
budget - 0.2831246739697444%
gross - 0.02464788732394366%
company - 0.002217005738132499%
runtime - 0.0005216484089723526%
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h4 id="Replace-nan-values-to-median-for-budget-and-gross">Replace nan values to median for budget and gross<a class="anchor-link" href="#Replace-nan-values-to-median-for-budget-and-gross">¶</a></h4>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># - Replace nan values to median for budget and gross</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">'</span><span class="se">\n</span><span class="s1">'</span><span class="p">,</span><span class="n">df</span><span class="p">[</span><span class="s1">'budget'</span><span class="p">]</span><span class="o">.</span><span class="n">describe</span><span class="p">())</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">'</span><span class="se">\n</span><span class="s1">Median of budget </span><span class="si">{</span><span class="n">df</span><span class="p">[</span><span class="s2">"budget"</span><span class="p">]</span><span class="o">.</span><span class="n">median</span><span class="p">()</span><span class="si">:</span><span class="s1">,.0f</span><span class="si">}</span><span class="s1">'</span><span class="p">)</span>
<span class="n">df</span><span class="p">[</span><span class="s1">'budget'</span><span class="p">]</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="s1">'budget'</span><span class="p">]</span><span class="o">.</span><span class="n">median</span><span class="p">(),</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">df</span><span class="p">[</span><span class="s1">'gross'</span><span class="p">]</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="s1">'gross'</span><span class="p">]</span><span class="o">.</span><span class="n">median</span><span class="p">(),</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>


<span class="k">for</span> <span class="n">col</span> <span class="ow">in</span> <span class="p">[</span><span class="s1">'budget'</span><span class="p">,</span> <span class="s1">'gross'</span><span class="p">]:</span>
    <span class="n">pct_missing</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="n">col</span><span class="p">]</span><span class="o">.</span><span class="n">isnull</span><span class="p">())</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">'</span><span class="si">{</span><span class="n">col</span><span class="si">}</span><span class="s1"> - </span><span class="si">{</span><span class="n">pct_missing</span><span class="si">}</span><span class="s1">%'</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>
 count    5.497000e+03
mean     3.558988e+07
std      4.145730e+07
min      3.000000e+03
25%      1.000000e+07
50%      2.050000e+07
75%      4.500000e+07
max      3.560000e+08
Name: budget, dtype: float64

Median of budget 20,500,000
budget - 0.0%
gross - 0.0%
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h4 id="Change-data-type-of-budget-and-gross">Change data type of budget and gross<a class="anchor-link" href="#Change-data-type-of-budget-and-gross">¶</a></h4>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># - Change data type of budget and gross</span>
<span class="n">df</span><span class="p">[</span><span class="s1">'budget'</span><span class="p">]</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s1">'budget'</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s1">'int64'</span><span class="p">)</span>
<span class="n">df</span><span class="p">[</span><span class="s1">'gross'</span><span class="p">]</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s1">'gross'</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s1">'int64'</span><span class="p">)</span>
<span class="n">df</span><span class="o">.</span><span class="n">dtypes</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[ ]:</div>
<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain" tabindex="0">
<pre>name         object
rating       object
genre        object
year          int64
released     object
score       float64
votes       float64
director     object
writer       object
star         object
country      object
budget        int64
gross         int64
company      object
runtime     float64
dtype: object</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">10</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[ ]:</div>
<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html" tabindex="0">
<div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>name</th>
<th>rating</th>
<th>genre</th>
<th>year</th>
<th>released</th>
<th>score</th>
<th>votes</th>
<th>director</th>
<th>writer</th>
<th>star</th>
<th>country</th>
<th>budget</th>
<th>gross</th>
<th>company</th>
<th>runtime</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>The Shining</td>
<td>R</td>
<td>Drama</td>
<td>1980</td>
<td>June 13, 1980 (United States)</td>
<td>8.4</td>
<td>927000.0</td>
<td>Stanley Kubrick</td>
<td>Stephen King</td>
<td>Jack Nicholson</td>
<td>United Kingdom</td>
<td>19000000</td>
<td>46998772</td>
<td>Warner Bros.</td>
<td>146.0</td>
</tr>
<tr>
<th>1</th>
<td>The Blue Lagoon</td>
<td>R</td>
<td>Adventure</td>
<td>1980</td>
<td>July 2, 1980 (United States)</td>
<td>5.8</td>
<td>65000.0</td>
<td>Randal Kleiser</td>
<td>Henry De Vere Stacpoole</td>
<td>Brooke Shields</td>
<td>United States</td>
<td>4500000</td>
<td>58853106</td>
<td>Columbia Pictures</td>
<td>104.0</td>
</tr>
<tr>
<th>2</th>
<td>Star Wars: Episode V - The Empire Strikes Back</td>
<td>PG</td>
<td>Action</td>
<td>1980</td>
<td>June 20, 1980 (United States)</td>
<td>8.7</td>
<td>1200000.0</td>
<td>Irvin Kershner</td>
<td>Leigh Brackett</td>
<td>Mark Hamill</td>
<td>United States</td>
<td>18000000</td>
<td>538375067</td>
<td>Lucasfilm</td>
<td>124.0</td>
</tr>
<tr>
<th>3</th>
<td>Airplane!</td>
<td>PG</td>
<td>Comedy</td>
<td>1980</td>
<td>July 2, 1980 (United States)</td>
<td>7.7</td>
<td>221000.0</td>
<td>Jim Abrahams</td>
<td>Jim Abrahams</td>
<td>Robert Hays</td>
<td>United States</td>
<td>3500000</td>
<td>83453539</td>
<td>Paramount Pictures</td>
<td>88.0</td>
</tr>
<tr>
<th>4</th>
<td>Caddyshack</td>
<td>R</td>
<td>Comedy</td>
<td>1980</td>
<td>July 25, 1980 (United States)</td>
<td>7.3</td>
<td>108000.0</td>
<td>Harold Ramis</td>
<td>Brian Doyle-Murray</td>
<td>Chevy Chase</td>
<td>United States</td>
<td>6000000</td>
<td>39846344</td>
<td>Orion Pictures</td>
<td>98.0</td>
</tr>
<tr>
<th>5</th>
<td>Friday the 13th</td>
<td>R</td>
<td>Horror</td>
<td>1980</td>
<td>May 9, 1980 (United States)</td>
<td>6.4</td>
<td>123000.0</td>
<td>Sean S. Cunningham</td>
<td>Victor Miller</td>
<td>Betsy Palmer</td>
<td>United States</td>
<td>550000</td>
<td>39754601</td>
<td>Paramount Pictures</td>
<td>95.0</td>
</tr>
<tr>
<th>6</th>
<td>The Blues Brothers</td>
<td>R</td>
<td>Action</td>
<td>1980</td>
<td>June 20, 1980 (United States)</td>
<td>7.9</td>
<td>188000.0</td>
<td>John Landis</td>
<td>Dan Aykroyd</td>
<td>John Belushi</td>
<td>United States</td>
<td>27000000</td>
<td>115229890</td>
<td>Universal Pictures</td>
<td>133.0</td>
</tr>
<tr>
<th>7</th>
<td>Raging Bull</td>
<td>R</td>
<td>Biography</td>
<td>1980</td>
<td>December 19, 1980 (United States)</td>
<td>8.2</td>
<td>330000.0</td>
<td>Martin Scorsese</td>
<td>Jake LaMotta</td>
<td>Robert De Niro</td>
<td>United States</td>
<td>18000000</td>
<td>23402427</td>
<td>Chartoff-Winkler Productions</td>
<td>129.0</td>
</tr>
<tr>
<th>8</th>
<td>Superman II</td>
<td>PG</td>
<td>Action</td>
<td>1980</td>
<td>June 19, 1981 (United States)</td>
<td>6.8</td>
<td>101000.0</td>
<td>Richard Lester</td>
<td>Jerry Siegel</td>
<td>Gene Hackman</td>
<td>United States</td>
<td>54000000</td>
<td>108185706</td>
<td>Dovemead Films</td>
<td>127.0</td>
</tr>
<tr>
<th>9</th>
<td>The Long Riders</td>
<td>R</td>
<td>Biography</td>
<td>1980</td>
<td>May 16, 1980 (United States)</td>
<td>7.0</td>
<td>10000.0</td>
<td>Walter Hill</td>
<td>Bill Bryden</td>
<td>David Carradine</td>
<td>United States</td>
<td>10000000</td>
<td>15795189</td>
<td>United Artists</td>
<td>100.0</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">re</span>
<span class="n">x</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s1">'released'</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">str</span><span class="p">)</span><span class="o">.</span><span class="n">str</span><span class="p">[:]</span>
<span class="n">y</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">x</span><span class="p">:</span>
    <span class="n">m</span> <span class="o">=</span> <span class="n">re</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="s1">'\d</span><span class="si">{4}</span><span class="s1">'</span><span class="p">,</span> <span class="n">i</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">m</span><span class="p">:</span>
        <span class="n">y</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">m</span><span class="o">.</span><span class="n">group</span><span class="p">())</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">y</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="kc">None</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">y</span><span class="p">[:</span><span class="mi">10</span><span class="p">])</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>['1980', '1980', '1980', '1980', '1980', '1980', '1980', '1980', '1981', '1980']
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="p">[</span><span class="s1">'yearcorrect'</span><span class="p">]</span> <span class="o">=</span> <span class="n">y</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[ ]:</div>
<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html" tabindex="0">
<div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>name</th>
<th>rating</th>
<th>genre</th>
<th>year</th>
<th>released</th>
<th>score</th>
<th>votes</th>
<th>director</th>
<th>writer</th>
<th>star</th>
<th>country</th>
<th>budget</th>
<th>gross</th>
<th>company</th>
<th>runtime</th>
<th>yearcorrect</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>The Shining</td>
<td>R</td>
<td>Drama</td>
<td>1980</td>
<td>June 13, 1980 (United States)</td>
<td>8.4</td>
<td>927000.0</td>
<td>Stanley Kubrick</td>
<td>Stephen King</td>
<td>Jack Nicholson</td>
<td>United Kingdom</td>
<td>19000000</td>
<td>46998772</td>
<td>Warner Bros.</td>
<td>146.0</td>
<td>1980</td>
</tr>
<tr>
<th>1</th>
<td>The Blue Lagoon</td>
<td>R</td>
<td>Adventure</td>
<td>1980</td>
<td>July 2, 1980 (United States)</td>
<td>5.8</td>
<td>65000.0</td>
<td>Randal Kleiser</td>
<td>Henry De Vere Stacpoole</td>
<td>Brooke Shields</td>
<td>United States</td>
<td>4500000</td>
<td>58853106</td>
<td>Columbia Pictures</td>
<td>104.0</td>
<td>1980</td>
</tr>
<tr>
<th>2</th>
<td>Star Wars: Episode V - The Empire Strikes Back</td>
<td>PG</td>
<td>Action</td>
<td>1980</td>
<td>June 20, 1980 (United States)</td>
<td>8.7</td>
<td>1200000.0</td>
<td>Irvin Kershner</td>
<td>Leigh Brackett</td>
<td>Mark Hamill</td>
<td>United States</td>
<td>18000000</td>
<td>538375067</td>
<td>Lucasfilm</td>
<td>124.0</td>
<td>1980</td>
</tr>
<tr>
<th>3</th>
<td>Airplane!</td>
<td>PG</td>
<td>Comedy</td>
<td>1980</td>
<td>July 2, 1980 (United States)</td>
<td>7.7</td>
<td>221000.0</td>
<td>Jim Abrahams</td>
<td>Jim Abrahams</td>
<td>Robert Hays</td>
<td>United States</td>
<td>3500000</td>
<td>83453539</td>
<td>Paramount Pictures</td>
<td>88.0</td>
<td>1980</td>
</tr>
<tr>
<th>4</th>
<td>Caddyshack</td>
<td>R</td>
<td>Comedy</td>
<td>1980</td>
<td>July 25, 1980 (United States)</td>
<td>7.3</td>
<td>108000.0</td>
<td>Harold Ramis</td>
<td>Brian Doyle-Murray</td>
<td>Chevy Chase</td>
<td>United States</td>
<td>6000000</td>
<td>39846344</td>
<td>Orion Pictures</td>
<td>98.0</td>
<td>1980</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h4 id="Order-by-Gross-column">Order by Gross column<a class="anchor-link" href="#Order-by-Gross-column">¶</a></h4>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Order by Gross column </span>
<span class="n">df</span><span class="o">.</span><span class="n">sort_values</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="p">[</span><span class="s1">'gross'</span><span class="p">],</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">ascending</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h4 id="Drop-any-duplicates">Drop any duplicates<a class="anchor-link" href="#Drop-any-duplicates">¶</a></h4>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Drop any duplicates</span>

<span class="n">df</span><span class="p">[</span><span class="s1">'company'</span><span class="p">]</span><span class="o">.</span><span class="n">drop_duplicates</span><span class="p">()</span><span class="o">.</span><span class="n">sort_values</span><span class="p">(</span><span class="n">ascending</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[ ]:</div>
<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain" tabindex="0">
<pre>7129                                thefyzz
5664                            micro_scope
6412               iDeal Partners Film Fund
4007                               i5 Films
6793                             i am OTHER
                       ...                 
3748                     1+2 Seisaku Iinkai
3024                        .406 Production
7525    "Weathering With You" Film Partners
4345        "DIA" Productions GmbH &amp; Co. KG
7657                                    NaN
Name: company, Length: 2386, dtype: object</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[ ]:</div>
<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html" tabindex="0">
<div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>name</th>
<th>rating</th>
<th>genre</th>
<th>year</th>
<th>released</th>
<th>score</th>
<th>votes</th>
<th>director</th>
<th>writer</th>
<th>star</th>
<th>country</th>
<th>budget</th>
<th>gross</th>
<th>company</th>
<th>runtime</th>
<th>yearcorrect</th>
</tr>
</thead>
<tbody>
<tr>
<th>5445</th>
<td>Avatar</td>
<td>PG-13</td>
<td>Action</td>
<td>2009</td>
<td>December 18, 2009 (United States)</td>
<td>7.8</td>
<td>1100000.0</td>
<td>James Cameron</td>
<td>James Cameron</td>
<td>Sam Worthington</td>
<td>United States</td>
<td>237000000</td>
<td>2847246203</td>
<td>Twentieth Century Fox</td>
<td>162.0</td>
<td>2009</td>
</tr>
<tr>
<th>7445</th>
<td>Avengers: Endgame</td>
<td>PG-13</td>
<td>Action</td>
<td>2019</td>
<td>April 26, 2019 (United States)</td>
<td>8.4</td>
<td>903000.0</td>
<td>Anthony Russo</td>
<td>Christopher Markus</td>
<td>Robert Downey Jr.</td>
<td>United States</td>
<td>356000000</td>
<td>2797501328</td>
<td>Marvel Studios</td>
<td>181.0</td>
<td>2019</td>
</tr>
<tr>
<th>3045</th>
<td>Titanic</td>
<td>PG-13</td>
<td>Drama</td>
<td>1997</td>
<td>December 19, 1997 (United States)</td>
<td>7.8</td>
<td>1100000.0</td>
<td>James Cameron</td>
<td>James Cameron</td>
<td>Leonardo DiCaprio</td>
<td>United States</td>
<td>200000000</td>
<td>2201647264</td>
<td>Twentieth Century Fox</td>
<td>194.0</td>
<td>1997</td>
</tr>
<tr>
<th>6663</th>
<td>Star Wars: Episode VII - The Force Awakens</td>
<td>PG-13</td>
<td>Action</td>
<td>2015</td>
<td>December 18, 2015 (United States)</td>
<td>7.8</td>
<td>876000.0</td>
<td>J.J. Abrams</td>
<td>Lawrence Kasdan</td>
<td>Daisy Ridley</td>
<td>United States</td>
<td>245000000</td>
<td>2069521700</td>
<td>Lucasfilm</td>
<td>138.0</td>
<td>2015</td>
</tr>
<tr>
<th>7244</th>
<td>Avengers: Infinity War</td>
<td>PG-13</td>
<td>Action</td>
<td>2018</td>
<td>April 27, 2018 (United States)</td>
<td>8.4</td>
<td>897000.0</td>
<td>Anthony Russo</td>
<td>Christopher Markus</td>
<td>Robert Downey Jr.</td>
<td>United States</td>
<td>321000000</td>
<td>2048359754</td>
<td>Marvel Studios</td>
<td>149.0</td>
<td>2018</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h4 id="Assuming-The-Most-correlated-fields-with-gross-are">Assuming The Most correlated fields with gross are<a class="anchor-link" href="#Assuming-The-Most-correlated-fields-with-gross-are">¶</a></h4><h4 id="--Budget-v/s-Gross">- Budget v/s Gross<a class="anchor-link" href="#--Budget-v/s-Gross">¶</a></h4><h4 id="--Company-v/s-Gross">- Company v/s Gross<a class="anchor-link" href="#--Company-v/s-Gross">¶</a></h4><h4 id="To-find-or-visualize-the-relation-between-budget-and-gross-is-by-scatter-plot">To find or visualize the relation between budget and gross is by scatter plot<a class="anchor-link" href="#To-find-or-visualize-the-relation-between-budget-and-gross-is-by-scatter-plot">¶</a></h4>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Assuming The Most correlated fields with gross are </span>
<span class="c1">#  - Budget v/s Gross</span>
<span class="c1">#  - Company v/s Gross</span>

<span class="c1"># To find or visualize the relation between budget and gross is by scatter plot</span>

<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="n">df</span><span class="p">[</span><span class="s1">'budget'</span><span class="p">],</span> <span class="n">y</span><span class="o">=</span><span class="n">df</span><span class="p">[</span><span class="s1">'gross'</span><span class="p">])</span>

<span class="c1"># Lets give some information</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">'Budget V/S Gross'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">'Budget of the Movie'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">'Gross of the Movie'</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA+0AAALCCAYAAACryjJEAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAADSKElEQVR4nOzdeZwU1bn/8W91zw4CguCoo6JRk0giiBHcEolZxRV3FA1umGgS470RlyxqbhJFzWJ+WW7c4kJE40KiglyNigYNYKKISlySSARhZAdhGJjpqt8fRTe91tbV3dXdn/fr5Qtnqrr7dFVNdz3nPOc5hmVZlgAAAAAAQOTEKt0AAAAAAACQH0E7AAAAAAARRdAOAAAAAEBEEbQDAAAAABBRBO0AAAAAAEQUQTsAAAAAABFF0A4AAAAAQEQRtAMAAAAAEFEE7QAAAAAARBRBOwAAZXTXXXfJMAzdddddlW4KAACoAgTtAIC6YRhGzn/Nzc0aOnSovvKVr+gf//hHpZtYUmPGjJFhGJ73X79+vfr06aPW1latXbvWcd8lS5YoHo9ryJAh2rp1a872YcOG6ZOf/GTq50Qiodtuu01HHnmkBg4cqMbGRg0ZMkQHHHCALrjgAj366KPe39g277//vr73ve/p0EMP1U477aTGxkb1799fBx54oC6++GLNnj3b93MCAFBpDZVuAAAA5XbNNdek/n/9+vWaP3++7rnnHj388MOaM2eORowYUbnGRUj//v116qmn6u6779a9996rb37zmwX3vfPOO2Wapr7yla+oqakpY9vbb7+tRYsW6fvf/74kO2A/9thjNWvWLA0YMEDHHHOMOjo6tHXrVr3xxhu677779Oabb+r444/33NZbb71Vl156qbq7u/Wxj31M48aN05AhQ7Rp0ya99dZbuueee/Sb3/xG3/72t3XTTTcFOyAAAFQAQTsAoO5ce+21Ob/7xje+oV/+8pf6+c9/Tup6mkmTJunuu+/W7bffXjBoN01Tv/vd71L7Z5s+fbokady4cZKkadOmadasWRo+fLiee+459e/fP2P/rq4uzZs3z3Mb77nnHl100UXacccddf/99+uEE07I2WfDhg367W9/q/fee8/z8wIAEAWkxwMAIOmLX/yiJGnlypUZv7/22mtlGEbe1OrFixfLMAxNnDgxZ9s///lPnXrqqdpxxx3Vp08fHXbYYZoxY4ZjG/7v//5Phx9+uPr06aOBAwfqxBNP1JtvvqmJEyfKMAwtXrw45zHz5s3TKaecovb2djU1NWn33XfXRRddpGXLluW087nnnpOUOU1gzJgxjm067LDDNGzYML322msFA+knn3xS//nPfzRmzBjtu+++OdunT5+uoUOHpjIYXnzxRUnSxIkTcwJ2SWpra9NnP/tZx3YlrV+/Xt/61rckSQ888EDegF2S+vXrp8svv1w/+9nPMn6ffn7vu+8+jR49Wn379tXQoUNT+yxfvlyXXHKJhg4dqqamJg0ePFgnnXSS/v73v+e8ztatW/WLX/xCI0eO1I477qi2tjYNHTpUJ5xwgv785z9n7PuXv/xFxx13nDo6OtTc3Kz29nYdcsghuu666zy9dwBAfWCkHQAAKRVQfepTnyr6ud555x0deuihWr16tY4++miNGDFC//znP3XiiSfq6KOPzvuY+++/X2eeeaZaWlp02mmnaZdddtGLL76oQw89VMOHD8/7mDvvvFOTJk1Sc3Ozjj/+eO2+++565513dPvtt+uxxx7T3Llztccee2jAgAG65pprdNddd+k///lPxvSA9OC0kAsvvFDf+ta3dPvtt2v06NE522+//fbUftnef/99zZ8/PxVYS9KgQYMk2WnzxXrooYe0du1aHXbYYfrCF77gun9DQ/5bn5/85Cd66qmndNxxx+mzn/2s1q9fL0l69913dcQRR2jZsmU66qijNH78eC1ZskQPPvigZsyYoYcffljHHnts6nkmTpyoadOm6ROf+ITOOecctba2atmyZZozZ45mzZqlz3/+85KkWbNm6ZhjjlG/fv10/PHHa7fddtOaNWv0j3/8Q7/+9a8zzhEAoM5ZAADUCUmWJOuaa65J/XfZZZdZRxxxhGUYhnXsscdaGzZsyHjMNddcY0mynn322Zzne/fddy1J1le+8pWM33/hC1+wJFk///nPM37/xz/+MdWG3/3ud6nfb9iwwRowYIDV1NRkLViwIOMxV1xxReox7777bur3b731ltXY2Gh95CMfsZYuXZrxmD//+c9WLBazTjzxxIzfH3nkkVaQr/41a9ZYLS0tVt++fa0PP/wwY9sHH3xgNTY2WoMGDbK6u7tzHvvLX/7SkmT95S9/Sf3u5ZdfthobGy3DMKwJEyZYDz/8sLV48WLf7bIsyzr33HMtSdZ3v/vdQI9Pnt+2tjbr5Zdfztn+xS9+0ZJk/fCHP8z4/QsvvGDF43Fr4MCBqWOybt06yzAM66CDDrJ6e3tznmvVqlWp/z/ppJMsSTnn27Isa+XKlYHeCwCgNjHSDgCoO/nSj/fff3+NHz9eO+ywQ1HPvXTpUj311FPaa6+99PWvfz1j2wknnKAjjzwylaae9Kc//Unr1q3TueeemzOq/t3vfle//e1vtW7duozf/+Y3v1FPT49uueUW7bbbbhnbPve5z+n444/XY489pg8//LDo97TjjjvqlFNO0dSpU3X//ffrggsuSG27++671dPTo3POOUfNzc05j50+fbqGDBmiww47LPW7Aw88UFOnTtWll16qqVOnaurUqZKkgQMH6jOf+YzOO+88HXfccZ7a1tnZKUk5x0CS1q1bp5///Oc5v89X02DSpEk68MADM363dOlSPfnkk9pjjz00efLkjG2HHXaYxo8fr6lTp+qRRx7ROeecI8MwZFmWmpubFYvlzkBMZhika21tzfndTjvtlPM7AED9ImgHANQdy7JS/79p0ya98cYbuvLKK3XWWWfpjTfe0I9+9KPAz/3KK69Iko444gjF4/Gc7WPGjMkJ2tMfk61v374aMWJEzpz6v/71r5Kk5557Ti+99FLO41asWKFEIqG3335bBx10UKD3km7SpEmaOnWqbrvttoyg3Sk1fu3atXruued07rnn5gSxp512msaNG6dnn31Wc+bM0SuvvKI5c+boj3/8o/74xz/qnHPOSa1pH9S6devydtDkC9pHjRqV87vkefn0pz+txsbGnO1HHXWUpk6dqldeeUXnnHOO+vXrp+OOO06PPfaYRowYoZNPPlmf/vSnNXr0aLW1tWU89qyzztIjjzyi0aNH6/TTT9dnP/tZHX744ero6Aj4bgEAtYqgPc2iRYv06KOP6t1339XatWv17W9/O++XuJMXX3xR06dP1/Lly9WvXz99+ctf9rVkDQCgvPr06aNRo0bpkUceUUdHh2688UZ99atf1e677x7o+ZJzoXfeeee829vb230/Jt/vV69eLUmuy5dt3LjRcbtXn/70p/Wxj31M8+fP12uvvaZPfvKTev755/X222/riCOO0Mc//vGcxzz66KPq7e1NVY3P1tjYqC9+8YupIoCJREIPP/ywzjvvPN1zzz0aN26cTjzxRMd2JY9neuG9pKFDh2Z00HR0dOj99993fJ50yfOyyy675H1M8vfpWRAPPPCApkyZovvuuy81L72lpUWnnHKKbr755tS5POmkk/T444/rJz/5ie6880799re/lSQddNBBuv766z3NzwcA1Aeqx6fZsmWLhg4dqvPPPz/Q41955RX9v//3//SFL3xBP/nJT3TBBRdoxowZmjVrVsgtBQCEbcCAAfroRz+q3t5evfzyy6nfJ0eIe3t7cx6TnbIuKVUN/YMPPsj7Osl07nT9+vVzfEy+3ydfZ/369bIsq+B/Rx55ZN7nDCI5mn7bbbdl/JtvmTfJTo3v16+fPve5z3l6/ng8rtNOO02XXXaZJOmZZ55xfczhhx8uSXr66ac9vUYh+Ub0k8c43zmT7Kry6ftJdrr7tddeq7ffflvvvfeepk6dqiOOOEJTp07VKaeckvH4Y445Rs8884zWrl2rp59+WpdddpneeOMNHXvssVq0aFFR7wcAUDsI2tMceOCBOuOMMwqOrvf09KTWgj377LN19dVX64033khtf/7553XwwQfri1/8onbeeWeNHDlSJ554ov70pz9l9PQDAKJp7dq1kux1x5N23HFHSdKSJUty9v/b3/6W87vkvOg5c+YokUjkbM+3dFz6Y7Jt3LhRCxYsyPn9IYccIsleNsyrZLp+vnZ58ZWvfEXNzc2aOnWqOjs79fDDD2vHHXfUqaeemrNvV1eXnnzySR1zzDFqamry9TrJOfhevjtPOeUUDRgwQC+++GLRgXu29POSr9Pm2WeflSSNHDky7+N33313nXXWWfq///s/7bPPPpozZ04qQyJdnz59dNRRR+mnP/2prr76am3dulVPPPFEiO8EAFDNCNp9uOOOO/TOO+/oW9/6lm666SYdcsgh+vGPf5zqae/p6cmZ89bU1KTVq1fnrPsLAIiWP/7xj3r33XfV2NiYUTQt2ZH7u9/9LiNwW7JkiX7wgx/kPE9HR4e+8IUv6N1339Uvf/nLjG1/+tOfcuazS3aBuv79++v3v/+9Xn311YxtP/zhD/OO6H/9619XY2OjLrvssrxLp23dujUnoE8WQnvvvfdy9vdi0KBBGjdunNauXavTTjtNmzdv1oQJE9TS0pKz76xZs7R58+a8qfHTpk3TU089ldE5ktTZ2Zkawf/MZz7j2qb+/funis2ddtppevzxx/Pu19XVpZ6eHtfnS5c8l4sXL84paDdv3jzdd9992nHHHVPvceXKlXrttddynmfTpk3auHGjGhoaUh0Yzz//fN6OgGRWRfYceABA/WJOu0erVq3S7Nmz9etf/1oDBw6UJB1//PF69dVX9eyzz+rMM8/UiBEjdPfdd+u1117TsGHD1NnZmbp5WLdunYYMGVLJtwAA2Ca9ENmmTZu0aNGi1Mjmj3/844w55KNHj9ZnPvMZPf/88xo1apSOOuooffDBB3rsscf0pS99Ke8I/K9+9Ssdeuih+ta3vqUnn3xSw4cP1z//+U9Nnz49VagsXb9+/fSrX/1KZ599tg477LCMddpfffXVVMX59GJuH/vYx3TnnXfqvPPO07Bhw/TlL39Z++23n3p6evTee+/pL3/5iwYPHqw333wz9ZjPfe5zevDBB3XSSSdp7Nixam1t1Z577qmzzz7b87GbNGmS7r///lSHQKHU+EceeUQtLS1516WfN2+ebrnlFrW3t+uII47QXnvtJcleE33GjBnavHmzTjjhhJx08kK+8pWvaMuWLfrmN7+p4447Th//+Md1+OGHa8iQIdq4cWOqCvzGjRs9dQSk+9///V8dfvjhuvzyy/Xkk0/qU5/6VGqd9lgspt/97nepzID3339fBx54oD75yU/qgAMO0O67764NGzbo8ccfV2dnp775zW+m9v3mN7+p999/X4cffriGDh2qpqYm/f3vf9czzzyjPffcU2eccYavdgIAalhlVpqLvlNPPdWaN29e6ue///3v1qmnnmpNmDAh478zzjjD+ulPf2pZlmWZpmnde++91plnnmmdfvrp1sSJE60//OEP1qmnnmq9/fbblXorAIBttG298/T/4vG41d7ebh1//PHWk08+mfdxa9eutS644AJr8ODBVlNTkzVs2DDrt7/9bcF12i3Lst555x3r5JNPtvr372+1tbVZhxxyiPX4449bv/vd73LWaU+aOXOmdeihh1qtra3WgAEDrOOPP976xz/+YR1zzDGWJGvt2rU5j1m4cKH1la98xdpjjz2spqYma8cdd7SGDRtmTZo0yXr66acz9u3t7bWuuuoqa6+99rIaGhosSdaRRx7p+zjuu+++liTr0EMPzbt969at1oABA6zjjjsu7/b33nvP+uUvf2mdeOKJ1n777WftsMMOVmNjo9Xe3m4dffTR1r333mslEgnf7Vqy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"/>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h4 id="Visualize-using-seaborn">Visualize using seaborn<a class="anchor-link" href="#Visualize-using-seaborn">¶</a></h4>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Visualize using seaborn </span>

<span class="n">sns</span><span class="o">.</span><span class="n">regplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="n">df</span><span class="p">[</span><span class="s1">'budget'</span><span class="p">],</span> <span class="n">y</span><span class="o">=</span><span class="n">df</span><span class="p">[</span><span class="s1">'gross'</span><span class="p">],</span> <span class="n">data</span><span class="o">=</span><span class="n">df</span><span class="p">,</span> <span class="n">scatter_kws</span><span class="o">=</span><span class="p">{</span><span class="s2">"color"</span><span class="p">:</span> <span class="s2">"red"</span><span class="p">},</span> <span class="n">line_kws</span><span class="o">=</span><span class="p">{</span><span class="s2">"color"</span><span class="p">:</span> <span class="s2">"blue"</span><span class="p">})</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[ ]:</div>
<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain" tabindex="0">
<pre>&lt;Axes: xlabel='budget', ylabel='gross'&gt;</pre>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA+0AAAK6CAYAAABMu73oAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAEAAElEQVR4nOz9eXQc93nn+3+qqquxEgDBDaSohSJECiJFcRFJW6IiL3Gc7djxZDyxLXlO5k48SUw5isdJxJkoXhLFIzmWb5SIubF99LuOHUUZRzOeKPGaG0eyYTskRVGCAFLcJIWiSJAEsQON7qqu+v1R6AYaaCwN9FLdeL/O4ZGAbgDf7qruruf7fb7PY/i+7wsAAAAAAISOWeoBAAAAAACA7AjaAQAAAAAIKYJ2AAAAAABCiqAdAAAAAICQImgHAAAAACCkCNoBAAAAAAgpgnYAAAAAAEKKoB0AAAAAgJAiaAcAAAAAIKQI2gEAAAAACKlIqQcQJsePH9czzzyj1157TX19ffqd3/kd7dmzJ6ff8eMf/1jf+MY3dPHiRTU0NOhnf/Zn9Z73vKdAIwYAAAAAVDJW2ieJx+O64YYb9J//839e0M8fO3ZMf/7nf653vetdevTRR/Vrv/Zr+uY3v6nvfOc7eR4pAAAAAGApYKV9kh07dmjHjh0z3u44jp566in96Ec/0ujoqK699lrdc8892rJliyTpBz/4gXbv3q2f+ZmfkSStWbNGv/RLv6S///u/17vf/W4ZhlGUxwEAAAAAqAystOfgiSee0OnTp/Xbv/3b+pM/+RO95S1v0Wc/+1ldvHhRUhDU27ad8TPRaFRXr17VlStXSjFkAAAAAEAZI2ifp56eHj377LP6+Mc/rra2NrW0tOg973mPbr75Zv3Lv/yLJGn79u06fPiwXn75ZXmepwsXLugf//EfJUn9/f0lHD0AAAAAoByRHj9P586dk+d5uv/++zO+77qu6uvrJUnvfOc71d3drYcffljJZFI1NTX6+Z//ef3d3/0dqfEAAAAAgJwRtM/T2NiYTNPUI488ItPMTFCorq6WJBmGoXvvvVcf+tCH1N/fr4aGBr388suSgv3tAAAAAADkgqB9nm644QZ5nqeBgQG1tbXNel/TNNXc3CxJ+tGPfqRNmzapoaGhGMMEAAAAAFQQgvZJxsbG1N3dnf768uXLev3111VfX69169Zp3759evzxx/Uf/+N/1IYNGzQ4OKiXX35Z119/vXbu3KnBwUH967/+q7Zs2SLHcfQv//Iv+slPfqLPfOYzJXxUAAAAAIByZfi+75d6EGHR1dWVNcC+++67tX//frmuq//9v/+3nnvuOfX29qqhoUE33XST/sN/+A+67rrrNDg4qEceeUTnzp2TJG3atEkf+MAHdNNNNxX7oQAAAAAAKgBBOwAAAAAAIUXLNwAAAAAAQoqgHQAAAACAkCJoBwAAAAAgpAjaAQAAAAAIKVq+jevr65PruqUexpxWrVqlK1eulHoYyDOOa+Xi2FYujm3l4thWJo5r5eLYVq5KPraRSETLly+f330LPJay4bquHMcp9TBmZRiGpGCsFP2vHBzXysWxrVwc28rFsa1MHNfKxbGtXBzbCaTHAwAAAAAQUgTtAAAAAACEFEE7AAAAAAAhRdAOAAAAAEBIEbQDAAAAABBSBO0AAAAAAIQUQTsAAAAAACFF0A4AAAAAQEgRtAMAAAAAEFIE7QAAAAAAhBRBOwAAAAAAIUXQDgAAAABASBG0AwAAAAAQUgTtAAAAAACEFEE7AAAAAAAhRdAOAAAAAEBIEbQDAAAAABBSBO0AAAAAAIQUQTsAAAAAACFF0A4AAAAAQEhFSj0AAAAAVDjPk93ZKbO3V15zs5ytWyWTtSMAmA+CdgAAABRMtL1d9QcPKnLmjAzHkW/bcltbNbx/vxL79pV6eAAQekxxAgAAoCCi7e1qOnBA9okT8uvqlFyzRn5dnewTJ9R04ICi7e2lHiIAhB5BOwAAAPLP81R/8KCM4WElW1rk19RIpim/pkbJlhYZw8OqP3hQ8rxSjxQAQo2gHQAAAHlnd3YqcuaMvKYmyTAybzQMeU1Nipw5I7uzsyTjA4ByQdAOAACAvDN7e4M97FVVWW/3q6pkuK7M3t4ijwwAygtBOwAAAPLOa26Wb9sy4vGstxvxuPxIRF5zc5FHBgDlhaAdAAAAeeds3Sq3tVVmf7/k+5k3+r7M/n65ra1B+zcAwIwI2gEAAJB/pqnh/fvl19fL6u6WEYtJnicjFpPV3S2/vl7D+/fTrx0A5sC7JAAAAAoisW+f+h9+WE5bm4yREVmXL8sYGZHT1qb+hx+mTzuA/PI82R0dqnr2WdkdHRXTnSJS6gEAAACgciX27VPvHXfI7uyU2dsrr7k5SIlnhR1AHkXb21V/8KAiZ84ERTBtW25rq4b37y/7CUKCdgAAABSWacrZtq3UowBQoaLt7Wo6cEDG8LC8piZ5VVUy4nHZJ06o6cCBss/sYYoTAAAAAFCePE/1Bw/KGB5WsqVFfk2NZJrya2qUbGmRMTys+oMHyzpVnqAdAAAAAFCW7M5ORc6ckdfUJBlG5o2GIa+pSZEzZ2R3dpZkfPlA0A4AAAAAKEtmb2+wh72qKuvtflWVDNeV2dtb5JHlD0E7AAAAAKAsec3N8m1bRjye9XYjHpcfichrbi7yyPKHoB0AAAAAUJacrVvltrbK7O+XfD/zRt+X2d8vt7U16FpRpgjaAQAAAADlyTQ1vH+//Pp6Wd3dMmIxyfNkxGKyurvl19dreP/+sm4zWb4jBwAAAAAseYl9+9T/8MNy2tpkjIzIunxZxsiInLa2sm/3JtGnHQAAAABQ5hL79qn3jjtkd3bK7O2V19wcpMSX8Qp7CkE7AAAAAKD8maacbdtKPYq8K/9pBwAAAAAAKhRBOwAAAAAAIUXQDgAAAABASBG0AwAAAAAQUgTtAAAAAACEFEE7AAAAAAAhRdAOAAAAAEBIEbQDAAAAABBSBO0AAAAAAIQUQTsAAAAAACFF0A4AAAAAQEgRtAMAAAAAEFIE7QAAAAAAhBRBOwAAAAAAIUXQDgAAAABASBG0AwAAAAAQUgTtAAAAAACEFEE7AAAAAAAhRdAOAAAAAEBIEbQDAAAAABBSBO0AAAAAAIQUQTsAAAAAACFF0A4AAAAAQEgRtAMAAAAAEFIE7QAAAAAAhBRBOwAAAAAAIUXQDgAAAABASBG0AwAAAAAQUgTtAAAAAACEFEE7AAAAAAAhRdAOAAAAAEBIEbQDAAAAABBSBO0AAAAAAIQUQTsAAAAAACFF0A4AAAAAQEgRtAMAAAAAEFIE7QAAAAAAhBRBOwAAAAAAIUXQDgAAAABASBG0AwAAAAAQUgTtAAAAAACEFEE7AAAAAAAhRdAOAAAAAEBIEbQDAAAAABBSBO0AAAAAAIQUQTsAAAAAACFF0A4AAAAAQEhFSj2Ayb7xjW/o8OHDevPNNxWNRrVp0ybde++9Wrdu3Yw/8+yzz+ov/uIvMr5n27aefPLJQg8XAAAAAICCClXQfvz4cb373e/Wxo0blUwm9dRTT+mhhx7SF77wBVVXV8/4czU1NXrssceKOFIAAAAAAAovVEH77//+72d8vX//fv3ar/2aXn31Vd1yyy0z/pxhGGpqairw6AAAAAAAKK5QBe1TjY6OSpLq6+tnvd/Y2Jg++tGPyvd9bdiwQR/84Ad17bXXZr2v4zhyHCf9tWEYqqmpSf9/mKXGF/ZxIjcc18rFsa1cHNvKxbGtTBzXysWxrVwc2wmG7/t+qQeRjed5+tznPqeRkRH90R/90Yz3O3XqlC5evKjrr79eo6OjeuaZZ3TixAl94Qtf0IoVK6bd/+tf/7qefvrp9NcbNmzQI488UpDHAAAAAADAYoQ2aP/yl7+sF198UX/4h3+YNfieieu6+vjHP64777xTH/jAB6bdPtNK+5UrV+S6bl7GXiiGYailpUXd3d0K6WHDAnBcKxfHtnJxbCsXx7YycVwrF8e2clX6sY1EIlq1atX87lvgsSzIE088oRdeeEGf+cxncgrYpeDBb9iwQd3d3Vlvt21btm1nva1cTgbf98tmrJg/jmvl4thWLo5t5eLYViaOa+Xi2FYujm3I+rT7vq8nnnhChw8f1ic/+UmtXr0659/heZ7OnTun5cuXF2CEAAAAAAAUT6hW2p944gm1t7fr937v91RTU6P+/n5JUm1traLRqCTp8ccfV3Nzsz70oQ9Jkp5++mnddNNNamlp0cjIiJ555hlduXJF73znO0v1MAAAAAAAyItQBe3f+973JEmf/vSnM77/0Y9+VG9729skST09PRkVBIeHh/XFL35R/f39qqur04033qiHHnpI69evL9awAQAAAAAoiFAF7V//+tfnvM/UgP5Xf/VX9au/+quFGRAAAAAAACUUqj3tAAAAAABgAkE7AAAAAAAhRdAOAAAAAEBIEbQDAAAAABBSBO0AAAAAAIQUQTsAAAAAACFF0A4AAAAAQEgRtAMAAAAAEFIE7QAAAAAAhBRBOwAAAAAAIUXQDgAAAABASBG0AwAAAAAQUgTtAAAAAACEFEE7AAAAAAAhRdAOAAAAAEBIEbQDAAAAABBSBO0AAAAAAIQUQTsAAAAAACFF0A4AAAAAQEgRtAMAAAAAEFIE7QAAAAAAhBRBOwAAAAAAIUXQDgAAAABASBG0AwAAAAAQUgTtAAAAAACEFEE7AAAAAAAhRdAOAAAAAEBIEbQDAAAAABBSBO0AAAAAAIQUQTsAAAAAACFF0A4AAAAAQEgRtAMAAAAAEFIE7QAAAAAAhBRBOwAAAAAAIUXQDgAAAABASBG0AwAAAAAQUgTtAAAAAACEFEE7AAAAAAAhRdAOAAAAAEBIEbQDAAAAABBSBO0AAAAAAIQUQTsAAAAAACFF0A4AAAAAQEgRtAMAAAAAEFIE7QAAAAAAhBRBOwAAAAAAIRUp9QAAAABCwfMU6eqSOjoUkeRs2SKZrG8AAEqLoB0AACx50fZ21R88qMjZs1IyqeWWJXfjRg3v36/Evn2lHh4AYAlj+hgAACxp0fZ2NR04IPvECfm1tdLatfJra2WfOKGmAwcUbW8v9RABAEsYQTsAAFi6PE/1Bw/KGB5WsqVFfk2NZJrya2qUbGmRMTys+oMHJc8r9UgBAEsUQTsAAFiy7M5ORc6ckdfUJBlG5o2GIa+pSZEzZ2R3dpZkfAAAELQDAIAly+ztleE48quqst7uV1XJcF2Zvb1FHhkAAAGCdgAAsGR5zc3ybVtGPJ71diMelx+JyGtuLvLIAAAIELQDAIAly9m6VW5rq8z+fsn3M2/0fZn9/XJbW+Vs3VqS8QEAQNAOAACWLtPU8P798uvrZXV3y4jFJM+TEYvJ6u6WX1+v4f376dcOACgZPoEAAMCSlti3T/0PPyynrU3G6Kh08aKM0VE5bW3qf/hh+rQDAEoqUuoBAAAAlFpi3z713nGH7K4urZLUJ8nZsoUVdgBAyRG0AwAASJJpyt22TVq7Vu7Fi9P3uAMAUAJMHwMAAAAAEFIE7QAAAAAAhBRBOwAAAAAAIUXQDgAAAABASBG0AwAAAAAQUgTtAAAAAACEFEE7AAAAAAAhRdAOAAAAAEBIEbQDAAAAABBSBO0AAAAAAIQUQTsAAAAAACFF0A4AAAAAQEgRtAMAAAAAEFIE7QAAAAAAhFSk1AMAAAAA5s3zZHd2yuztldfcLGfrVslkHQpA5SJoBwAAQFmItrer/uBBRc6ckeE48m1bbmurhvfvV2LfvlIPDwAKgmlJAAAAhF60vV1NBw7IPnFCfl2dkmvWyK+rk33ihJoOHFC0vb3UQwSAgiBoBwAAQLh5nuoPHpQxPKxkS4v8mhrJNOXX1CjZ0iJjeFj1Bw9KnlfqkQJA3hG0AwAAINTszk5FzpyR19QkGUbmjYYhr6lJkTNnZHd2lmR8wKJ4nuyODlU9+6zsjg4mnzANe9oBAAAQamZvrwzHkVdVlfV2v6pK5sCAzN7eIo8MWBzqNGA+WGkHAABAqHnNzfJtW0Y8nvV2Ix6XH4nIa24u8siAhaNOA+aLoB0AAACh5mzdKre1VWZ/v+T7mTf6vsz+frmtrUH7N6AcUKcBOSBoBwAAQLiZpob375dfXy+ru1tGLCZ5noxYTFZ3t/z6eg3v30+/dpQN6jQgF7yzAQAAIPQS+/ap/+GH5bS1yRgZkXX5soyRETltbep/+GH2/6KspOo0+LPUaTBclzoNkEQhOgAAAJSJxL596r3jDtmdnTJ7e+U1Nwcp8aywo8xMrtPg19RMu506DZiMoB0AAADlwzTlbNtW6lEAi5Kq02CfOKFkdXVmivx4nQanrY06DZBEejwAAAAAFBd1GpADzgIAAAAAKDLqNGC+QpUe/41vfEOHDx/Wm2++qWg0qk2bNunee+/VunXrZv25n/zkJ/qf//N/6sqVK2ppadE999yjnTt3FmnUAAAAAJA76jRgPkIVtB8/flzvfve7tXHjRiWTST311FN66KGH9IUvfEHV1dVZf+bkyZN67LHH9KEPfUg7d+5Ue3u7/uRP/kSPPPKIrrvuuiI/AgAAAADIAXUaMIdQTeH8/u//vt72trfp2muv1Q033KD9+/erp6dHr7766ow/861vfUvbt2/Xe97zHq1fv14f+MAHdOONN+o73/lOEUcOAAAAAED+hWqlfarR0VFJUn19/Yz3OXXqlH7xF38x43u33Xabjhw5kvX+juPIcZz014ZhqGa8zYIxuWpjCKXGF/ZxIjcc18rFsa1cHNvKxbGtTBzXysWxrVwc2wmhDdo9z9NXvvIVbd68edY09/7+fjU2NmZ8r7GxUf39/Vnv/41vfENPP/10+usNGzbokUce0apVq/Iy7mJoaWkp9RBQABzXysWxrVwc28rFsa1MHNfKxbGtXBzbEAftTzzxhN544w394R/+YV5/7/ve976MlfnUzM2VK1fkum5e/1a+GYahlpYWdXd3y/f9Ug8HecJxrVwc28rFsa1cHNvKxHGtXBzbylXpxzYSicx74TiUQfsTTzyhF154QZ/5zGe0YsWKWe/b1NSkgYGBjO8NDAyoqakp6/1t25Zt21lvK5eTwff9shkr5o/jWrk4tpWLY1u5OLaVieNauTi2lYtjG7JCdL7v64knntDhw4f1yU9+UqtXr57zZzZt2qSXX34543sdHR266aabCjVMAAAAAACKIlRB+xNPPKEf/vCHuv/++1VTU6P+/n719/crkUik7/P444/rb/7mb9Jf//zP/7xeeukl/cM//IPefPNNff3rX9fZs2f1sz/7s6V4CAAAAAAA5E2o0uO/973vSZI+/elPZ3z/ox/9qN72trdJknp6ejIqCG7evFm/9Vu/pb/927/VU089pbVr1+p3f/d36dEOAAAAACh7oQrav/71r895n6kBvSS99a1v1Vvf+tYCjAgAAAAAgNIJVXo8AAAAAACYQNAOAAAAAEBIEbQDAAAAABBSBO0AAAAAAIQUQTsAAAAAACFF0A4AAAAAQEgRtAMAAAAAEFIE7QAAAAAAhBRBOwAAAAAAIUXQDgAAAABASBG0AwAAAAAQUgTtAAAAAACEFEE7AAAAAAAhRdAOAAAAAEBIEbQDAAAAABBSBO0AAAAAAIQUQTsAAAAAACFF0A4AAAAAQEgRtAMAAAAAEFIE7QAAAAAAhBRBOwAAAAAAIUXQDgAAAABASBG0AwAAAAAQUgTtAAAAAACEFEE7AAAAAAAhRdAOAAAAAEBIEbQDAAAAABBSBO0AAAAAAIQUQTsAAAAAACFF0A4AAAAAQEgRtAMAAAAAEFIE7QAAAAAAhBRBOwAAAAAAIUXQDgAAAABASBG0AwAAAAAQUgTtAAAAAACEFEE7AAAAAAAhRdAOAAAAAEBIEbQDAAAAABBSBO0AAAAAAIQUQTsAAAAAACFF0A4AAAAAQEgRtAMAAAAAEFIE7QAAAAAAhBRBOwAAAAAAIUXQDgAAAABASBG0AwAAAAAQUgTtAAAAAACEFEE7AAAAAAAhRdAOAAAAAEBIEbQDAAAAABBSBO0AAAAAAIRUpNQDAACgrHieIl1dUkeHIpKcLVskkzlwYMnyPNmdnTJ7e+U1N8vZupX3BAB5RdAOAMA8RdvbVX/woCJnz0rJpJZbltyNGzW8f78S+/aVengAiiz9nnDmjAzHkW/bcltbeU8AkFdMAwIAMA/R9nY1HTgg+8QJ+bW10tq18mtrZZ84oaYDBxRtby/1EAEUUcZ7Ql2dkmvWyK+r4z0BQN4RtAMAMBfPU/3BgzKGh5VsaZFfUyOZpvyaGiVbWmQMD6v+4EHJ80o9UgDFwHsCgCIiaAcAYA52Z6ciZ87Ia2qSDCPzRsOQ19SkyJkzsjs7SzI+AMXFewKAYiJoBwBgDmZvb7Bftaoq6+1+VZUM15XZ21vkkQEoBd4TABQTQTsAAHPwmpvl27aMeDzr7UY8Lj8SkdfcXOSRASgF3hMAFBNBOwAAc3C2bpXb2iqzv1/y/cwbfV9mf7/c1tag1ROAisd7AoBiImgHAGAupqnh/fvl19fL6u6WEYtJnicjFpPV3S2/vl7D+/fTmxlYKnhPAFBEvJMAADAPiX371P/ww3La2mSMjkoXL8oYHZXT1qb+hx+mJzOwxGS8J4yMyLp8WcbICO8JAPIuUuoBAABQLhL79qn3jjtkd3VplaQ+Sc6WLaymAUtU+j2hs1Nmb6+85uYgJZ73BAB5RNAOAEAuTFPutm3S2rVyL16cvp8VwNJimnK2bSv1KABUMKYBAQAAAAAIKYJ2AAAAAABCiqAdAAAAAICQImgHAAAAACCkCNoBAAAAAAgpgnYAAAAAAEKKoB0AAAAAgJAiaAcAAAAAIKQI2gEAAAAACCmCdgAAAAAAQoqgHQAAAACAkCJoBwAAAAAgpAjaAQAAAAAIKYJ2AAAAAABCiqAdAAAAAICQImgHAAAAACCkCNoBAAAAAAgpgnYAAAAAAEIqUuoBAAAAzIvnye7slNnbK6+5Wc7WrZLJ+oMknhsAqGAE7QAAIPSi7e2qP3hQkTNnZDiOfNuW29qq4f37ldi3r9TDKymeGwCobEzBAgCAUIu2t6vpwAHZJ07Ir6tTcs0a+XV1sk+cUNOBA4q2t5d6iCXDcwMAlY+gHQAAhJfnqf7gQRnDw0q2tMivqZFMU35NjZItLTKGh1V/8KDkeaUeafHx3ADAkhCq9Pjjx4/rmWee0Wuvvaa+vj79zu/8jvbs2TPj/bu6uvSZz3xm2ve/9KUvqampqYAjBQAAxWB3dipy5oy8pibJMDJvNAx5TU2KnDkju7NTzrZtJRljqfDcAMDSEKqgPR6P64YbbtA73vEOff7zn5/3z/3pn/6pamtr0183NDQUYngAAKDIzN5eGY4jr6oq6+1+VZXMgQGZvb1FHlnp8dwAwNIQqqB9x44d2rFjR84/19jYqLq6ugKMCAAAlJLX3CzftmXE40H69xRGPC4/EpHX3FyC0ZUWzw0ALA0LDtp7enrU09Ojm2++Of29119/Xf/4j/8ox3F05513zprank+/93u/J8dxdO211+r9739/xpimchxHjuOkvzYMQzXjH3TG1NSykEmNL+zjRG44rpWLY1u5OLbF4956q5KtrYqcOKFkdXVmGrjvy+zvl9vWJvfWW/NyPMrp2Bb7uSln5XRckRuObeXi2E4wfN/3F/KDn/vc5xSPx/UHf/AHkqT+/n59/OMfl+u6qqmp0cDAgP7rf/2v2rt374IG9h/+w3+Yc0/7hQsX1NXVpY0bN8pxHP3zP/+zfvjDH+qP//iPdeONN2b9ma9//et6+umn019v2LBBjzzyyILGCAAAiuD735d+/deloSGpuVmqrpbGxqTeXqmhQfrLv5Te8Y5Sj7I0eG4AoOIteKX97Nmz+rmf+7n01z/4wQ+USCT06KOPavXq1frsZz+rf/iHf1hw0D4f69at07p169Jfb968WZcuXdI3v/lNfexjH8v6M+973/v0i7/4i+mvUzM3V65ckeu6BRtrPhiGoZaWFnV3d2uBcy0IIY5r5eLYVi6ObZG1tSn62c+q/vHHZZ09K+PqVfm2reTmzRq+7z4l2tqkixfz8qfK7tgW8bkpZ2V3XDFvHNvKVenHNhKJaNWqVfO770L/yPDwsBobG9NfHz16VLfccotaWlokSXv27NFTTz210F+/YK2trXrllVdmvN22bdm2nfW2cjkZfN8vm7Fi/jiulYtjW7k4tsUTv/NOxd/6VtmdnTJ7e+U1N8vZulUyTakAx6Ccjm2xn5tyVk7HFbnh2FYuju0igvaGhgZduXJFkjQyMqLTp0/rQx/6UPp2z/PklaAv6Ouvv67ly5cX/e8CAIACM01al82E5wYAKtaCg/Zbb71V3/72t1VbW6uuri75vp+x//z8+fNasWJFTr9zbGxM3d3d6a8vX76s119/XfX19Vq5cqX+5m/+Rr29vbrvvvskSd/85je1evVqXXvttUokEvr+97+vzs5OPfjggwt9WAAAAAAAhMaCg/YPfehDunjxor72ta8pEonowx/+sFavXi0pqND+k5/8RHfeeWdOv/Ps2bP6zGc+k/76q1/9qiTp7rvv1v79+9XX16eenp707a7r6qtf/ap6e3tVVVWl66+/Xn/wB3+grVu3LvRhAQAwO89TpKtL6uhQRJKzZUuQhgwAAFAAC64enzI6OqpoNKpIZCL+TyQSunDhglauXKn6+vpFD7IYrly5ktEKLowMw9DatWt18eLFJb+vo5JwXCsXx7byRNvbVX/woCJnzyqSTMq1LLkbN2p4/34l9u0r9fCQB7xuKxPHtXJxbCtXpR9b27bnXYhu0UsDtbW1GQG7JEWjUd1www1lE7ADADCXaHu7mg4ckH3ihPzaWmntWvm1tbJPnFDTgQOKtreXeogAAKACLThof/nll/XMM89kfO/73/++fvM3f1Mf+chH9JWvfKUkhegAAMg7z1P9wYMyhoeVbGmRX1Mjmab8mholW1pkDA+r/uBBic89AACQZwsO2v/u7/5Or7/+evrrc+fO6ctf/rIaGhp0yy236Nvf/va0oB4AgHJkd3YqcuaMvKYmyTAybzQMeU1Nipw5I7uzsyTjAwAAlWvBQfubb76pjRs3pr/+wQ9+oJqaGv3hH/6hPv7xj+ud73ynfvCDH+RlkAAAlJLZ2yvDceRXVWW93a+qkuG6Mnt7izwyAABQ6RYctI+Njammpib99Ysvvqjt27eravyCprW1Nd3HHQCAcuY1N8u3bRnxeNbbjXhcfiQir7m5yCMDAACVbsFB+8qVK3X27FlJUnd3t9544w1t27Ytffvw8LBs2178CAEAKDFn61a5ra0y+/ulqRVsfV9mf7/c1lY5tBwFAAB5tuA+7fv27dPTTz+t3t5enT9/XnV1ddq9e3f69ldffVVr167NyyABACgp09Tw/v1qOnBAVnd3sLfdsmTEYjL7++XX12t4/376tQMAgLxbcND+7/7dv5Prujp27JhWrlypj370o6qrq5MUrLJ3dXXp53/+5/M2UAAASimxb5/6H3443addQ0MyLEtOWxt92oEw8zxFurqkjg5FJDlbtjDBBqCsGH4ldqpfgCtXrshxnFIPY1aGYWjt2rW6ePGiOGyVg+NauTi2FcrzZHd1aZWkKyIAqDS8bitLtL09PdEWSSblWpbcjRuZaKsgvGYrV6UfW9u2tWrVqnndNy9XGWNjYzp//rzOnz+vsbGxfPxKAADCyTTlbtsmvfvdwX8J2IFQira3q+nAAdknTsivrZXWrpVfWyv7xAk1HTigaHt7qYcIAPOy4PR4STpz5oyefPJJvfLKK/I8T5JkmqZuvvlm3XvvvRkt4QAAAICi8DzVHzwoY3hYyZYWyTAk05RfU6NkdbWs7m7VHzyo3jvuYOINQOgtOGg/ffq0Pv3pTysSiegd73iHrrnmGklB//Yf/ehH+tSnPqVPf/rTam1tzdtgAQAAgLnYnZ2KnDkTFI00jMwbDUNeU5MiZ87I7uyUM6n7EQCE0YKD9r/9279Vc3Oz/uiP/khNTU0Zt73//e/XH/zBH+ipp57SH/zBHyx2jAAAAMC8mb29MhxHXlVV1tv9qiqZAwMye3uLPDIAyN2C84FOnz6td73rXdMCdklqamrST//0T+v06dOLGRsAAACQM6+5Wb5ty4jHs95uxOPyIxF5zc1FHhkA5G7BQbthGEomkzPe7nmejKnpSAAAAECBOVu3ym1tldnfL02tOu37Mvv75ba2ytm6tSTjA4BcLDho37x5s7773e/qypUr027r6enR9773Pd18882LGhwAAACQM9PU8P798uvrZXV3y4jFJM+TEYvJ6u6WX1+v4f37KUIHoCwseE/7Bz/4QX3yk5/Ub//2b2vPnj1au3atJOnChQt6/vnnZVmWPvjBD+ZtoAAAAMB8JfbtU//DD6f7tGtoSIZlyWlro087gLKy4KB9w4YN+h//43/oqaee0vPPP69EIiFJikaj2r59uz7wgQ9o/fr1eRsoAAAAkIvEvn3qveMO2V1dWiWpT5KzZQsr7ADKyoKCdsdx9NJLL2nVqlX63d/9XXmep8HBQUlSQ0ODTN4IAQBAufE8Rbq6pI4ORURwVzFMU+62bdLatXIvXpy+xx0AQm5BQXskEtEXvvAF/eqv/qquv/56maaZtYo8AGAJ8DzZnZ0ye3vlNTcHhZ0IdFBmou3tE2nUyaSWW5bcjRtJowYAlNyCgnbDMLR27VoNDQ3lezwAgDKSDnTOnJHhOPJtW25rK4EOykq0vV1NBw7IGB6W19Qk1dfLHx6WfeKEmg4cUP/DD3M+AwBKZsFLIe973/v0ne98RxcuXMjneAAAZSIV6NgnTsivq1NyzRr5dXXpQCfa3l7qIQJz8zzVHzwoY3hYyZYW+TU1kmnKr6lRsqVFxvCw6g8elDyv1CMFUG48T3ZHh6qefVZ2RwfvI1iwBReiO3XqlJYtW6ZPfOITuuWWW7Rq1SpFo9GM+xiGof/0n/7TogcJAAiZKYGODEOSgkCnulpWd7fqDx5U7x13kCqPULM7OxU5cyZYYR8/j9MMQ15TkyJnzsju7JSzbVtJxgig/JCJhnxacND+3e9+N/3/nZ2dM96PoB0AKg+BDiqF2dsrw3HkVVVlvd2vqpI5MCCzt7fIIwNQrqZuufGqqmTE42y5wYItOGj/n//zf+ZzHACAMkKgg0rhNTfLt20Z8XiQGj+FEY/Lj0TkNTeXYHQAyg6ZaCgAzhQAQM4mBzrZEOiEFPsrp3G2bpXb2iqzv396KzDfl9nfL7e1NeiKAABzyCUTDZivBa+0/8qv/Mqc94lGo2pubtaWLVv0nve8Ry0tLQv9cwCAEEkFOvaJE0pWV2demIwHOk5bG4FOiLC/cgamqeH9+9V04ICs7u7gQtuyZMRiMvv75dfXa3j/flbEAMwLmWgohAV/Av3yL/9yukf7rl279Au/8Av6hV/4Be3cuVOmaeqGG27Qz/zMz2j9+vV69tln9cADD+j111/P49ABACUzHuj49fWyurtlxGKS58mIxWR1dxPohAyV/meX2LdP/Q8/LKetTcboqHTxoozRUTltbew9BZATMtFQCAteaW9ubtbQ0JD+9E//VGvWrMm4rbu7W5/+9Ke1fv16ffjDH9bFixf14IMP6qmnntJ/+2//bdGDBgCUXirQSa3emgMD8iMROW1trN6GCfsr5yWxb59677hDdleXVknqk+Rs2bKknxMAuSMTDYWw4KD9mWee0bvf/e5pAbsktbS06N3vfrf+z//5P3r729+utWvX6l3veldGxXkAQPlLBzqdnTJ7e+U1NwcXIgQ6oUGl/xyYptxt26S1a+VevDh9jzsAzCXLlht/vHo8W26wUAsO2q9evSpzlpPNsiz19PSkv161apUcx1nonwMAhJVpEuyFGPsrAaC4yERDvi04aL/22mv1T//0T/qpn/opNTU1ZdzW39+v733ve7r22mvT37t06dK0+wEAUHY8T5GuLqmjQxGFP4WalmYAUHxkoiGfFhy0f/jDH9ZnP/tZ/dZv/ZZ2796drgzf3d2tI0eOKJlM6jd/8zclSYlEQs8995y2b9+el0EDAFAK6QrsZ89KyaSWW5bcjRtDvXLC/koAKBEy0ZAnCw7at2zZooceekhf//rXdfjwYSUSCUmSbdu69dZb9f73v1833nijpKD12xe/+MX8jBgAgBJIVWA3hoeD/eH19fKHh9MV2ENbZZz9lQAAlLUFB+2StGHDBj3wwAPyPE+Dg4OSpIaGhln3ugMAUHayVWA3zcJUYPe8vKdTsr8SAIDytaigPcU0TfarAwAqVrEqsKfT78+ckeE48m1bbmtrXgJr9lcCAFCe8hK0AwBQyYpRgX1q+r03nsKe1/R79lcCAFB2mF4HAGAOkyuwZ7PoCuxT0u/9mpqJ9PuWFhnDw6o/eFDyvEU8CgAAUI4I2gEAmEOqArvZ3y/5fuaN4xXY3dbWBVdgzyX9HgXkeYp0dEjf/W7wXyZJAAAhQHo8AABzyVKBXZYlIxbLSwX2YqTfY3bl2M4PADDB86TRUUOjo4ZWr66sSVdW2gEAmIdUBXanrU3G6Kh08aKM0VE5bW2L3m9e8PR7zCpVT8A+cUJ+ba20dq382tp0PYFoe3uph4jFIIMCqFiOI/X3G7pwwdTrr1u6dMnU0JAx9w+WGVbaAQCYp3QF9q4urZLUJ8nZsmXRFdhT6ff2iRNKVldnpsiPp987bW0LTr/HLIrZzg9FRwYFUFl8Xxobk0ZHTY2OGkokSj2i4uDTBwCAXJim3G3bpHe/O/hvPgK58fR7v75eVne3jFhM8jwZsZis7u5Fp99jZtQTqFxkUACVwfOk4WFDly6Z+rd/s3ThgqX+/qUTsEsE7QAAhEJG+v3IiKzLl2WMjOQl/R4zS9UT8GepJ2C4LvUEyg0dGYCylkgEae9vvKF02vvwsKFkstQjKw3S4wEACIl0+n1np8zeXnnNzUFKPCvsBTO5noBfXR1kOUgyJPk1NdQTKFO5ZFA427aVZIwAJgRp70a6kFwiIRmGIcua3rRlKSJoBwAgTEyTIKKI0vUEXnpJRjIZFAP0fUUMQ35VlXzLknPbbdQTKDN0ZADCL5mcqPY+OmqQ+DILgnYAALB0mabid9+tqh//WEom5UciMiIR+cmkjJERGZal+N13k+1QZjIyKGpqpt1OBgVQGomENDISBOljY5VX5b1Q+AQCAABLl+ep6rnn5NXVya+tlSFJrhukx9fWyqurU9Vzz7H3ucykMijM/v7pubXjHRnc1lYyKIAC830pFjN09aqpc+csvfGGpd5ek4A9R6y0AwCAJSu993n16mBP+9iYbEmulP6avc9laLwjQ9OBA7K6u4O97ZYlIxaT2d9PRwaggEh7zz+CdgAAsGRl7H02jCCV2rblO44k9j6Xs1RHhnSf9qEhGZYlp62NPu1AnsXjE4E6q+j5R9AOAACWLPY+V7Z0R4auLq2S1CfJ2bKFFXZgkVJp76lAfXyeEwVC0A4AAJasdPX4EyeUrK7ObA82vvfZaWtj73M5M02527ZJa9fKvXiR/lHAAqXS3kdGDMVipL0XE9OMAABg6Rrf++zX18vq7g76tHuejFhMVnc3e58BLGnxuNTXZ+jNNy29/rqly5dNjYyEN2Dv7TX13e9W6777mnTqVOWsT1fOIwEAAFgA9j4DQGBy2vvIiCHXLfWIZue6UmenrcOHq3ToUFSnT9vp27ZudbRpU8gfwDwRtAMAgCWPvc8AlirXnSgiVw5p7xcumDpyJAjSX3ghqtHR7O/Tzz5brd/4jZEij64wCNoBFJbnye7slNnbK6+5OdgXykUwgDBi7zOAJSIe1/jedFNjY6UezezGxqRjx6I6dKhKhw9H9cYbs4ewq1Yl9a53jemnfzpepBEWHkE7gIKJtrcH6aZnzshwHPm2Lbe1lXRTAEDxeJ4iXV1SR4ciIoMCS1Mq7X1kJFhRD3Pau+9Lr79u6fDhKr34onTkyColEjO3kYtGfd12W0J79iS0d29c11+fVGtrsngDLgKCdgAFEW1vV9OBAzKGh+U1NcmrqpIRj8s+cUJNBw6o/+GHyyNwJ1MAxcY5B+RNevL47FkpmdRyy5K7cSOTx1gSUmnvqWrvYU4eGhoydPRoVIcPByvqV65Yk26dHrBfd52rPXvi2rs3odtuS6i6unhjLQWCdgD553mqP3hQxvCwki0t6RZKfk2NktXVsrq7VX/woHrvuCPUwQiZAig2zjkgf6ZOHqu+Xv7wcPlNHgM5GBtL7U83FQ9xdngyKZ08GUnvTT9xwlYyOfNqem2tp127gtX0PXviWrs25Bvv84ygHUDe2Z2dipw5E1wkGVPegA1DXlOTImfOyO7slLNtW0nGOJeKyRRA2eCcA/Io2+SxaZbd5DEwF9+fKCIX9rT3nh5TR45EdfhwlZ5/PqqBgdlfe5s2OXrHO2xt3dqnW25JKLKEI9cl/NABFIrZ2yvDceRVVWW93a+qkjkwILO3t8gjm6cKyRRAGZl8zq1ZI2NsTMbIiGRZSq5ZI+vSJc45IAeVMHkMzMR1ld6bHua0d8eRXn45aMd2+HBUZ87Ys96/qcnTnj1x7dmT0O7dcTU3S6tXr9bly05oH2OxELQDyDuvuVm+bcuIx+XX1Ey73YjH5Uci8pqbSzC6uXGxh2JLnXN+NKrIa6/JiMclzwtWBquqOOcWivoAS1bZTx4DUwRp76ZGR41Qp71fuGCN70sP2rHFYjO/51qWry1bHO3dGwTqN93kTnmLnjldfqkhaAeQd87WrXJbW2WfOKFkdXVm4Ov7Mvv75bS1BRfQIcTFHorN7O2VMTIS/Juc2+h5MlxXZjwuv76ecy4H1AdY2sp98hjwvKDae9jT3mMx6cUXJ9qxnT8/e3i5Zk0yXUBu586E6uuX+BL6PBG0A8g/09Tw/v1qOnBAVne3vKYm+eP7c83+fvn19Rrevz+0K15c7KHYvKam6QH7JIbrSqliWpgT9QFQ7pPHWJrKIe3d96VXX43o8OGg0ntHR1SOM3s7th07Etq9O2jHdt11yWlJjJgbQTuAgkjs26f+hx9Or3SZAwPyIxE5bW2hX+niYg9F53kykpN6yk455yQFt3tLq1rugiymJgX9vCtHlsljWZaMWKwsJo+xdJRD2vvgoKHnn4+m96b39Fiz3v+GG9z03vTbbktohsRF5ICgHUDBJPbtU+8dd5TfntIyzxRA+bFfemne93N27izwaMrbQmtSFLyfN/vriy5j8vjsWWloSIZllcXkMSpXKu09taI+eb42LJJJ6ZVX7PRq+okTtjxv5uXx+vrMdmxr1jDBnG8E7QAKyzTLsnBWOWcKoEwZhnzLkuF5weq670+0qTLN4PuY00JqUhS6nzf760snPXnc1aVVkvpEBgWKz3GCtmwjI4bGxsKZ9t7TY44H6VU6ciSqoaGZXyOG4WvzZje9N72tzVnS7diKgacXAGZQtpkCKDvOjh1BHQXXlR+NSr4vw/flG4ZkGOlAz9mxo9RDDb2ca1IUuJ83++tDwDTlbtsmrV0r9+JFhTJiQsWJxaRYzNTIiKFEotSjmS6RyGzHdvbs7O3Yli9Pjq+kB+3Ympp4HRUTQTsAzKZMMwVQXpxt2+Ru2iT7+PEgQI9E5JtmELw7jiTJ3bSJc3Eecq1JUdAWj4vZXw+grHie0pXew5r2fv68pUOHojpypEovvBDV2NjMKe+W5evWW5303vTW1qnt2FBMBO1APrFnEcBCmKYGH3xQTfffL+vq1aDo3KT0+OSKFRp88EHeT+Yjx5oUhWzxWNAJAQAl5zgT1d7DmPY+Omro2DE73Y7twoXZQ7+WlmS6Z/rOnQnV1YXsAc0hEpFqanxVV5fXuOeDoB3IE/YsAliMxL596n/sMdU//rjsV14JchejUTk336zh++7jfSQHudSkKGSLx0JOCCAHS60rAAsIBRWLTVR7D1vau+9LZ88G7dgOHarSyy/bct2ZV9OrqoJ2bKlAff368mrHZppBkJ76F42WekSFQ9AO5EHR9izyQQxUNOoo5M98n8uMdPqqKhljY5IkQ5JfXb2oFo+FnBDA/BS8K0DIsICQf8nkRLX3WCx8ae8DA5nt2K5enb0d24YNjvbuDfam33prebVjMwypunoiSK+uLvWIioegHVisIu1Z5IMYWCKoo5A/83kux9Ppl99/v+yTJ9NFyiJSkMK+YsWCWzzmur8e+VXorgBhQ9HD/EkkJqq9x+PhSnt33Yl2bIcOVemVVyLy/dnbse3eHRSP27MnodWry6cTiWFINTXS8uWeqqs9TX0bXUoI2oFFKsaeRT6IAaCwfGmi1V6qnkDq+wuV4/76oqr0zK0CdwUIHYoeLorvS2NjRjpQH6//GRqXLwft2I4cCdqxDQ/P3o6trc1JV3q/+ebyasdWVTU55V265hrp4kU/VBMnpVBGhxAIp4LvWeSDGAAKJ/UeG4ulA2pJkmEEX8dii3qPzWV/fbEshcytpVYEcKk93nxIJjOrvXshWoCOx6WOjmi6b/prr80esjU3B+3Y9u6N6/bbE2psLJ8INxrVpHR3X9ak7H5jqS6rZ0HQDixSofcs8kEMAIVjd3bK7uqSOToqJZPyLUuGacr3fRmxmAzLkt3Vtaj32DDVKlgqmVtLrQjgUnu8CxWPTwTqs7U7Kzbfn2jHduhQlV58Map4fObxRSK+tm1LjAfqCd14o1s2aeOpCu+pf+WUBVBKPE3AIhV6zyIfxAAqQkjTsc2eHpmDg0pVlzJcV/L9YIXHMKRkUubgoMyenkX+oRDUKlhCmVtLrQjgUnu885VKe0+1ZQtT2vvIiKEXXoim96Z3d89eQO6aa9zxlPe4duxwVFtbHqvplpW5kl7JFd4LiaAdWKwC71nkgxhAuQtzOrbZ1xcE7JNzYw1jYm/75PuVuaWUuVXIrgBhRNHDCam091S197CkvXuedObMRDu2zk5byeTMy+M1NZ527HC0Z89EO7ZyYJqZFd7LqTp9mBG0A3lQyD2LfBADKGdhT8f2GhszgnOlVtilie/7fnC/MrekMrcK2BUglMJc9LAI4nGpr8/QyIip8fmZUOjvN3TkSNV4EbmoentnX03fuNFJr6bfeqtTFqvSqTZsqUB9KVd4LySCdiBPCrZnsVAfxCFNVc1QDmMEMLMySMc2BwYmVtZTppYpNozgfmVuKWZuZRzJfHUFCKkwFj0sFN+f3Dvd1NCQ1Nsb1KIoJdeVjh+3dehQUOn95MnZ27EtWxa0Y0utpq9cGZK0gDlUV2emvBOkFx5BO5BPBdqzmO8P4jCnqqaUwxjLChMgKIFySMf2li9XulzxlJT49Kq7ZQX3K3NLKnMrNWGUTMrZvDnI7pDkKsgosC5dKvmEUSGEqehhvrluZtp76qVa6oDx0iVThw9X6dChqF54YfZ2bKY50Y5t7964Nm92M6qlh1VVVWbKewWcTmWHoB0oE/n6IA57qmq5jLGcMAGSZ56nSFeX1NGhiCRny5aKuCAuhHJIx/ZWrpTX0BDsWZ+6Sje+Mus1NMhbubI0A8ynJZRCnTFhNN6fXbYtf7wSWRgmjAomDEUP82RsLFXt3VSqG2OpxePSSy9NFJD7t3+bPZxauTKZTnm//faEGhrCn+dh25kV3sthYqHSEbQD5WSxH8RlkKpaFmMsI0yA5Fd6AuTsWSmZ1HLLkrtxIxMgKVMyOrymptCnYztbtyq5bt3MEwe+r+S6dZWx+qylk0JdDhNGmM73M3unu26pRxSM6dw5S4cOVenIkaiOHYsqkZh5ed+2J7dji2vDhmTJswHmEolkrqTbdqlHhKkI2oElpBxSVcthjGWDCZC8mjoBovp6+cPDTICMy5rRsXGjvBUrZHV3hzod2xgenr7KnuL7we0VpJJTqFOW4v79cuW6Srdkm5z2XkrDw0E7ttTe9Lnasa1f76aD9O3bE8pyyoWKaWaupJdDwbuljqAdWELKYeWhHMZYLoo2AbIU9stnmwAZT7llAmSWjI5XXpFvWZJlZU/HrqtT7Od+TlU/+EHJzh27o0PWm28Gf3dStfj0a8YwZL35puyODjnbtxd1bAVVQSnU2Syp/ftlKEh7NzU6aoQi7d3zpNOnIzp0KKj03tU1dzu2nTsT42nvCV1zTbjbsaUqvE8Ujyv1iJArgnZgCSmHlYdyGGO5KMYEyFLZL08GyCzmkdGRbGkJnqOzZ9Pp2MmWFknSsj//85KeO/axYzJcV34kEhSk87ygl7cUBPLJpAzXlX3sWGUF7ZUuy/59WZaMWKzi9u+XA8+bqPY+OmooGYIYt68vaMeWWk3v75/9XLjpJke7dwer6Vu3OqFOITcMqaoqM0gPe4o+ZheqoP348eN65pln9Nprr6mvr0+/8zu/oz179sz6M11dXfrqV7+qN954QytWrNAv//Iv621ve1txBgyUmXJYeSiHMZaLQk+ALKX98mSAzGw+Exrm1asaePhhyTRl9vbKOndO9V/8ooyRkfCdO6Y5vQUcylLG/v2zZ6WhIRmWVXH798PKcSaqvY+NlT7t3XWlzk5bhw8Hq+mnTs0edTc2err99iBI3707oRUrwt2Oraoqsw0b81GVJVRBezwe1w033KB3vOMd+vznPz/n/S9fvqyHH35Y73rXu/Sxj31MnZ2d+su//Es1NTVpO7PhwHTlUDm4HMZYJgo6AbLE9suTATKzeU9o9Pcr/ra3SZ6n5nvukTEyEopzx9mxIzi2rhuk8k9hJJPybVvOjh0FHwvyL71/v6tLqyT1iY4PhRSLSbGYqZERQ4lEqUcjXbxopoP0o0ejGh2dvR3bLbc42rs3qPS+aVO427FFo0oH6FR4r3yhCtp37NihHTl8KH7ve9/T6tWr9R//43+UJK1fv16vvPKKvvnNbxK0AzMoh8rB5TDGslDACZClli5OBsjMcp3QCNu542zbJnfTJtnHj8tIJII0edMM0uTHS1e7mzZVxHm8ZJmm3G3bpLVr5V68SBZFHnleZrX3Uqe9j40F7dgOHYrq8OEqnTs3e6izalUyXUBu166Eli0L77kRiWQWj4uEKopDoZX14T59+rRuvfXWjO/ddttt+spXvlKaAQFlohwqB5fDGMtBoSZAlly6OPtjZ5TrhEbozh3T1OCDD6rp/vtlXb0qI5mUXFfGeLHB5IoVGnzwwSV5bIFsEomJQL3Uae++L73+uqVvflP6539u1Esvzd6OLRoN2rGlVtNvuCG87dgsKzPdnQrvS1tZB+39/f1qbGzM+F5jY6NisZgSiYSiWc5ux3HkOE76a8MwVDO+MmCE9VU7LjW+sI8zK89TZFLw5RJ8pZXsuFqW3NtumxhHcf/6/JTDGGcRltesc9dd6rvzzqyvwYWOzFuxQn40Ovvqqm3LW7Gi5I8/X5y77tLAI4+o/vHHZU3aH+u2tWn4vvvk7NtXdudoXliWRu67T40PPDBjRsfIfffJGM/dDOO549x1lwb+7M9U/+d/rsgrr8hyXXmRiNybb9bwxz62dI9tBQnL+3E58v2JtPfR0elp78V+SoeGDB09mlpNj+ry5VReePaJwOuuS7VjS2j79sSUyunhOR9MM7PCe+a8ZnjGWUy8bieUddC+EN/4xjf09NNPp7/esGGDHnnkEa1ataqEo8pNy3jF3bLx/e9LDz8snTwZTM9Go9LmzdKBA9I73lHq0YVG2R3XxfA86dgxqadHWrlS2rGjoidxQnNsr7kmf79rzRrpllukjg5p2bJpq6saHJS2bdOqd72rso7t+98v/fIvp8/fyMqViuzYoapKeowL8f73SytWSA8/LOvkSWloKHiv375dOnBAKya/14f13JlybK2VK2VxbCtOaN6PQy6ZlEZGgn+jo8HHdjSqkqz2ep7U2Sm1t0s//KH00kuaNQ2/rk5661ulffuku+6S1q+PKAh5aos15HkJ2rBJtbXBPyq8z4zXbZkH7U1NTRoYGMj43sDAgGpqarKuskvS+973Pv3iL/5i+uvUzM2VK1fkju9dCyvDMNTS0qLu7m75ZbIfK9rersYHHkhXl/YbGoLVlxdflP9rv6aBRx5Z8vuTy/G4Lka0vT29Uplq85TcuFHD991XcedCpR/b6Ec+Ery+33gj6+rqwEc+osSlS6UeZkEY11yjll27gmNboY9RUm5ZUm1t0v/7/2a//8WLGXcN87mzZI7tElPp78f5kEgo3ZJtbKy00ePVq6aOHAlW0g8fjmpgYPbJsy1bpJ07R7VnT9CObfJ+78uXCzzYHExuw1ZTEwTpiYRCUbQvjCr9dRuJROa9cFzWQftNN92kY8eOZXyvo6NDmzZtmvFnbNuWPUNjxXI5GXzfL4+xep7qHn981urSdY8/rvhb31pZK3ELVDbHdRGi7e1qzNIiLHLihBofeKC4bZ48r2j75Sv12MbvvHP2/fJ33lnxBZ8q9dhK4xNs48d23n3UDUPOlFoz2c6Bcjh3KvnYSirqe2DJeZ4iXV3SSy/JEtXjU3xfGhub6J0+afeopOKe+46T2Y7t9Om527Ht2RPXnt1x3bHiFbU2OuqV5Nx0k2SUvr1cSqrCe+rf1NMuLOMMu4p/P56HUAXtY2Nj6u7uTn99+fJlvf7666qvr9fKlSv1N3/zN+rt7dV9990nSfqZn/kZffe739Vf//Vf6+1vf7s6Ozv1k5/8RAcOHCjVQ8AkYasQjBILUYuwBQUjyIqCgZUp2t6upiwTbPnso865UzpL6T0w/VjPnpWSSS23LLkbN1bkY52PZDKz2rtXwtbjFy5YOnw42Jv+wgtRxWIzv/Yty9eWLU660vtNN7mqPnZUtU8+KevcOcnz1GCaSl53nUbvuUeJXbuK+Egm2HZm8TgqvCNfQnUqnT17Vp/5zGfSX3/1q1+VJN19993av3+/+vr61NPTk7599erVOnDggP7qr/5K3/rWt7RixQr9xm/8Bu3eQiJ0FYJRUmGZxClGMLLkmCYTb5WkmBNsnDtFt5TeA6c+VtXXyx8ersjHOpt4PLPae6nEYtKLLwat2A4diur8+dnDkDVrksFq+p6Edu1KqL5+YqU1evSolj36qIyREXkNDVJtrfzRUUXOntWyRx/V0Cc+UZTAfXKF95oaXzMk8wKLFqqgfcuWLfr6178+4+379+/P+jOf+9znCjksLFCuvXtR2UIxibOQYGQppZACCs8EGwogRBlPBZftsZpmZT7WKYJq7xOr6Zlp78Udx6uvRtL70js6onKc2duxbd+e0J49QTu266+foR2b76v2ySdljIwouWrVxLGtrlayqkrWlSuqffJJJXbuzHtlN9PMDNJpw4ZiCVXQjsqSa+9eVLYwTOLkGoykiuZFTpyYSCEdb++1FFZnMIPU/tiODkVUeftjQzHBhoJYShMyS+mxSpLrTqymx2KlS3sfHDT0/PPR9N70nh5r1vtff72rvXuD1fTbbktohredDPapU7LOnQtW2LMd24YGWefOyT51Ss7mzYt4NKkK7/6klPdF/TpgwQjaUTimqeH9+9V04MCMvXuH9++vqItdzCwMkzi5BCPR9nYtv/9+mVevZlSKsf71X2WfPq2+xx4jcF+ClsL+2DBMsKEwltKETMZj9X0ZsZikoNu1X1NTEY81Hg+qvcdipsbGSjOGZFI6eTKiQ4eCIP3ECVueN/Pqdl2dp127gp7pe/bEtWZN7rMLxsBAcGwbG7Pe7kejMoeGZEzpMDWv322kKrwrvS+dNmwIA4J2FFRi377ZKwRXyEUu5iEEkzjzDkaamtR44IDMy5eDKrSRSPBJ7vsyXFfm5ctqeOgh9XzrW0w6LSHp/bFDQ/JqaqTqavmJhOzjxytqf2wYJthQGEtpQib1WM2+PpkDAzLiccn3FTEM+VVV8hoby+6xptLeU9XeS9WpuKcnaMd26FCVnn8+qsHB2T8Hb77ZSe9Nv+UWZ9HF2fzGxuA8TiTkZ1n6NhIJ+ZGI/BmC+qmqqjKLx/GxjjAiaEfBUSEYKaWexJlvMCLPU+TkySBgt+2J+41/bTiOIidPyu7okEPhy6UhtT+2t1dGMqnIwEBGACDHqZz9sSGYYENhLKUJGWfrVnkrVsju6kpPvhqmKd/zZMRiskZH5WzZEvrHmkp7D1bUS9PKLJGQXn55oh3b2bOzV1tbvjw5vi89od2742pqyu+gnU2blLzuOkXOnlWyqmr6eTw4KHfjRjkztICORjNT3q3ZM/iBUCBoR3FQIRjjSjqJM89gxH7pJRmuO7HCPplhyLcsGa4r+9gxgvYlwu7slN3VJXN0VPK8aQGAYZqyu7oqZn9sqSfYUCBMyEwIcc7z2Fhqf7qpeLw0Yzh/3hovIFelY8fsOduxbd3qaO/euHbvTuimm9zCnkKGodF77tGyRx+VdeVKunq8MTYmc3BQfl2dRu+5J32MI5HM4nG0YUM54rQFUHwlnMSZTzASOX26JGNDeJk9PTIHB4OAPZV9kapYPJ59YQ4OypzUlrTckSVVmZbKhIzd2Snz6lW569bJ6u8P0uOTSRmGEVSQb2yUefVqKCbaPC+z2nsp0t5HRw0dOzaxmv7mm7OHCC0tyXQBuZ07E6qrK24KQGLXLg194hMTfdpHRmSYptyNGzX24Q/J3rdTDTUeFd5RMQjagbChxVjBzRWMODt2BIGY68rPkjdnJJPybVvOjh3FHjpKxOzrCwJ208yefWGaMjwvuF8lIUuqIi2FCZl0Ibo1a+QuXy5jbEy2JFcK9kH7vszLl0tWiM5xMqu9FzvtPdWO7dChYG/6yy/bct2Zsw+qqlLt2OJ6y1sSWr9+hnZsRZTYtUuJnTsVPXta66NJjVYlVbNjs1bUmJJKVD4fKBCCdiBE0pWpz5yZaDHW2lpRqx+hMUsw4mzbJnfTJtnHj6cL2kwuRCdJ7qZNBDNLiLd8uTQemGe7tjY8TzLN4H5AOajwCZmpRff8mhrJtuWPNy03xsaKXoguFpNiMVOjo0ZJ0t4HBjLbsV29Ovtm7g0b3HQBuW3b5teOrRgmKryP/2tt1bp1a3Xx4kX5pdj0DxQBQTsQEtH2djUeOCBjeFheU5O88X2G9okTFVWZulyMfOhDavjc54K2MalcxfF06OSKFRp88MGKWpXC7LyVK+U1NARVqB0nyMAwTcnzZCST6d7A3sqVpR4qAIWj6J7nTaymj44aSiYL9qeySialEyfs8b3pUb3yyuzt2OrrU+3Ygr3pC2nHVijV1ZkV3icfzlKv+APFQNAOhIHnqf7xx2UMDyvZ0pL+BPJrapSsrpbV3V05lalDbnK2g3w/KDrnefKiUam2Vs7NN2v4vvuYQFlinK1b5WzZEhQpdBwZiYTkOEExumg02C5RBpWogSUjS9E9WZaMWGxxRffm2MLmOEq3ZBsbK37a+5UrZrqA3PPPRzU0NPPjMwxfN988sZre1rb4dmz5Eo1mFo/j0gdLXUhemsASd+yYrLNng4uKLPtlvaYmRc6cCUXBnEqW7sOdynZobg6qKvf0SNGohn77tzV6771MnCxF4wHA8vvvlzE6GmwITW2ZcBz5DQ1Lp+o2UCYyiu6dPSsNDcmwrAUX3ZtpC9uVj/yW+m+7U6OjhhKJAj2YGcTj0ssvR3XoULCa/tprs7dja24O2rHt3RvXrl2JvLdjWyjbzlxJD8vkARAWvCSAMOjpCQrmzLBhzK+qkjkwULKCORVp6mrJLbcEfbizZTusXy+ru1vV3/52ELRjyUpf3o4H7OnzpGQjAjCbdNG9ri6tktQnydmyJecJtsmTum7jcg3bTRoZi2isMyb3v/2FRj9Ro8SuXQV5DJP5/kQ7tkOHqvTii1GNjc2cHx6J+Lr1Vie9mt7a6oYindyyMlfS7dnnGoAlj6AdCIOVKzMK5kxlxONFL5hTybKtlnhr1sg6dy54jrNlOzQ2KnL8uGr/6q/k7No1d6Vlz1Okq0vq6FBEC7tIRIh4XjCpk0zK2bw5qDeh8UrUVVWyLl2qzC0sdLNAGOV6Xpqm3G3bpLVr5V68qJxz1j1P0T//ovoHLQ2svE1jqpbvGVJU0kpf1pUrqn3ySSV27izIBuvRUUMvvBAdD9Sjunhx9sv3tWtd7d2bSLdjq60t/bSiaSq9il5T44emqB1QLgjagTDYsUPJjRsVKWHBnKIIQQAwLQV+vOCf9eqrMoeG5C1bJk2ZODGGh2VduiQjFtOyP/kT+XV1s1b1T08KnD0rJZNabllyN26kC0AZszs7FTlzJtjCYprTKlFX4hYWulkgjBZ0Xi5gEtX3pbGxYG964oXTqjmVkF+/Qb6qM+84XoTSOndO9qlTcjZvXvRj9H3pzJmgHduRI3O3Y6uu9rVjRyK9mh6GdmyGoXSAHgTrpR0PUO4I2oEwME0N33efGh94IF0wxx8PJhdVMCdEQhEApFZLs6TAe6tWyRwaknX5stxly9K3GcPDss6fDyqEm2ZQHdw0Z6zqP3VSQPX18oeHi9sFIASTI5Um3fN5iWxhmWlyi24WKKWFnJe5TKImk5nV3r3x4unRnsHg9d/YmHVcfjQadBoZGFjwY+vvN3TkSNCK7ciRqHp7Z2/HduONzvje9IRuvTWhaHTBfzpvZqvwDmBxCNqBkMgomHPmjMyBAfmRyIIL5oRJWAKAjNXSKVcTfk1NMFEyNiYjFpNfWyv5vqzLl4OAXZJfXR183zCyV/XPNikwvipbrC4AoZgcmUsZTipM7fk8VUVtYZllcqviu1mwrSW8FnBezmcSdWj3vnSQPtPecL+xMXj9JxLysywZG4mE/EhE/gxBfTauG7RjCwrIVenkyYh8f+Yod9kyT7ffPtGObdWq0rdjq6rKDNJ5qQCFQ9AOhEi6YE6ZBTSzClEAMOtqqWEouWaNIm+8IfPKFXmrVwc9uMfGgtstS8nVqyeC/SxV/TMmBaSgyrgkY/zxFjqFOiyTI3ONMfSTClmEoedzsWQ9j5NJybKKch6XCttawm22SdesXVZmmET1qms0srpJse5hDXz+f+nK539qzn3ozqZNSl53nSJnzypZVTX99T84KHfjRjmbNs36ey5dMnXkSJUOHYrq6NGohodn/swzTV9tbY527w5W02++2ZE1++J7wU2u8F5T45d8PMBSQtAOhI1pVtSFcM4XWgU012qpbFteU5Pc664L9rCPjEieF0wwrFkjv74+4+5TU6JTkwK+4yhy8aKMeFzyfUUMQ35VlZIrV8pw3cKkUIdocmQm5TCpMKNC9XwOoZnOY83nPC7DLAopJNtaMKtct6hM/uxJytJIskYJv0H9iYg8mTLqx2T824X57UM3DI3ec4+WPfqorCtX5DU0yI9GZSQSMgcH5dfVafSee6Z9xsXjUkdHNL2a/vrrs192r1iRHC8gF9fttyfU0FDaAnKRSOa+dCq8A6VD0A6goMK0F3heq6Vbtqj3a1+Tffy47KNHteyxx4IaA/NIifaam4P02jffDIL9SESGacr3PBmxmCJvvimvoaEgKdRhmhzJqgwmFeaS757PYZXtPE73pJ/lPC7XLIowbGvB3HLdouJc6ldfvE4D9ddrLBHcPxqJyJMrKfd96IlduzT0iU+o9sknZZ07J3NoSH4kInfjRo3ec48Su3bJ96Vz54J2bIcPB+3Y4vGZV/Ft29ettybSe9NvvLG07dhSFd5T/8KwTx5AgKAdQEGFai9wltXSrAX/IhE527bJ2bpV1d/73rxTop1bbgnSiF1XfiqFMhUAjD8HSiaD++X7oYVociSb0E8qzFO+ej6HWdbzWApW2mc4j8s5i6JSzs1KN9ekq9HXr4HNO3SxZZtG/s2S4axTg7VGftzQ1ILv0sL2oSd27VJi507Zp07JGBiQ39io/ms26+gLVTr8+SBQ7+6ePWf8mmvc9Gr69u1OSduxUeEdKB8E7QAKKmx7gXMq+DffIH88YLOPH5csK0ibdl35lpUuUGeM7wmWZck+fjzvF/+hmhzJIuyTCjlZbM/nuZQ4xTzreZxaac92Hpd5FkXGuTmeTSBN1KIoq3OzkmV5P3aitRqNmYr1Oxqu3arBf3+/EkPjQXOe9qFP5fmGOrVFh09GdehQlbq6bCWTMy+P19R42rHD0d69QTu2a65JLuTR54VhSFVVvmpqRIV3oMwQtAMorBwD32LIpeBfLkG+2dsbBHTr18vq6UmvSBqGIb+6WsmVK2WOjhbk4j9skyNThX1SISzCkGKe9TxO7WnPch6X+0p16tw0+/pkDgxMq0XhNTZyboZEYt8+df/h52T8xV8pfrZbcTciPxJR8sbr0inqaVn2oau2VsbY2Kz70LPp68tsx9bXN/tqemtr0I5tz564br3VKelecCq8A5WBoB1AwYWynV0OBf/mG+SnLv5l23I3bJAxNiZbkqugXZwxNla4i/98T47kebU37JMKYRCWFPNs57HhukEqcZbzuNyzKJytW+WtWCG7qyuYmJhSi8IaHZWzZcuSPjdLyfcze6e7198lPbwvI0Xd2bQpa/A9dR+6RkZkmGbGPvRsXFfq6rLTe9NPnpw96m5o8LR7dxCk796d0MqVpWvHRoV3oDIRtAMoirJvZzePID8jMG1pCVaUbVu+4xQlMM3X5EhBVnuLkXFRyLTyKb/bvfXW/PzeSb8/LCnm2c7j9AaALOdxxWdRkD9cdK4rjYwEQXosZkzfgWIYc1d8H5feh376tJolDUpybrpp2nHt7jZ1+HDQju2FF6IaGZm7HVtqb/rmzW7JguNIJDNIj3BlD1QkXtoAkC8haAu22MmRQq72FjLjopBp5dl+d7K1VfrUp6S2tkX97pRQpZjnOMFS7lkUdmenzKtX5a5bJ6u/P3NbS02Nko2NMq9eDW16f6UYG5NGR02NjhqKx/P8yw1D7ubN0urVci9flnxf8bj04otRHT4c7E0/d272S+JVq5LplPfbb09o2bLSFJCjwjuwNBG0AyiKMOzVLYZQtAXLIfU/QxFWewuRcVHIiYaZfnfkxAnp139d0c9+VvE771zw2FOyFkMbL/pWimJoCy7YePGivJqadAFGMxaTv2xZqHvYp5/7NWvkLl8+bVuLfF/m5cuhTe8vV54nxWJGekU9WeD6bL4vnTkjffvbNTp0KKqXXooqkZi9HdtttyW0e3dCb3lLXDfckCxJ0gUV3gFIBO0AiiAse3WLpVzbghVttXehkwrZFHKiYa7fffmy6h9/XPG3vnXRx3amYmgqYTG0xL596n3LW1Tz938v6803lbzmGsXe+15ly79N7NunkXvvVf2f/7kiFy6kx+4tW6aRe+8N9et7anp/xrYWqbC1KJYYxwn2p4+MGBoby5L2nmdDQ4aOHo2m96ZfvixJy2a8/7XXuunV9O3bE8qy26PgJiq8TwTp7NAAQNAOoLBCtFe3qArdFqwAFlxQrIQtygo50TDX71Zzs6yzZ/OSNp2tGFq6zVqJiqFly46pefrprBkj0fZ21f31X0uRiNxrrglSy31fZiymur/+azlbt4Y2cC/39P6wi8WkWMzUyIihRKKwf8vzpFOnIjp0KKj0fvz47O3Yams97dqV0J49Ce3eHde6daUpIEeFdwBzIWgHUFCh2quLWS2koNiCtj3kMcgvZOXyuX63qqtlXL1anLTpIi+15ZQdM3libu3aYNJh/PckGxvDPzEXgloUZWGer1vPy6z2Xui096tXTR05Eh1vx1algYHZj9NNN00UkNu61SlJ4bZoNDPlnQrvAOZC0I5ACVfKUNnKvR3UUpLriuNCtj3ku7ZBISuXz/W7NTYm37bzkjadtRhaKj2+2MXQ5psd85a3yD5+XPbRo4ocP76wibmQfPak9/A//rgir7wi9fTIiETk3Hyzhu+7L7RZAsUy1+s2kZgI1Aud9u44qXZsQaX306dnb8fW2Ohp9+649u519PM/3yDP65Nf5MwnKrwDWCzeNrBkCoShNCq+HVSpFCLYyaVq+AK2PRSitkEhU5vn+t3q7VVy8+a8pE1nFENrapIxMCAzkZAXjcpvbAzuU6RiaPPJjrG7urTyl35J5qVLMkZGZA4NyY/FlFyzRn59fcaPzDQxF8rPHt+X4TiS68rw/bLY1lJo2V63GovLOf66nN/9nLp/t1Yjt+0u6BguXDB15EgQpB89GlUsNvN7nWX52rLFSfdN37zZlWlKhmFo5cqG8X3thWVZmSvpVHgHsFgE7UvcUisQVtFCsmI1FftF86+Qwc58q4bnvO2hULUNCtn/fY7fraYmDd93X15eZzMVojMNQ35fX1EL0c25LcBxZPb3y3BdJVevll9TI3NkRMbYmKzz55Vcvz4jcJ9pW0WYPnui7e1afv/9Mq9eTQfqpqSqQ4dknzmjvsceW5qfhZNet4k16zTi12okWaMRq0ZesyHryhXZf/WU9Ojted3CEYul2rEFe9PfeGP2y9U1a5LasyeuPXsS2rmz+O3YTDMzSJ/ppQMAC0XQvpQt1QJhFSiUK1YphQyqlqBiBDvzacuW67aHQtY2KGT/95l+t9vWJutTn1KirS0vq7FhKkQ3a3aM78sa73OdXLUquN335VdXy4jFZCSTsi5flltXlx7/tIm5sH32eJ4aHnpI5uXL6efeME35nifDdWVevqyGhx5Sz7e+teTep/xjXRo81aPhups16jRl3mhIXkODrHPnZJ86JWfz5oX/HV967TUrHaS/9FJUjjPzJEA0GrRjS+1Nv/764rZjo8I7gGIjaF/CKBBWGcK2YpVNIYOqJaWYwc4cbdly3fZQ6NoGhej/Ptvvdm+9VWuvuUa6eHHRv39e8hURzCMjZ7bsGCMWC1qgVVVNHHfDUHL1akXOn5eSSRljYzJGRyXTzDoxF7bPHrujQ5GTJ4OA3baDMRmGZJrBOe44ipw8KbujQ8727QUfTyn5vjQ2ZqTbshmnEmqIL1NyWUP2+0ejMoeGZAwM5Py3hoYMPf/8xGr6lSuzV2O77jpXe/fGtXdvQrfdlij6ajYV3gGUEkH7EkaBsAoQthWrWRQyqFoqwhTs5LrtoSi1DfLZ/32O323keVmtGIXo5p2RM1t2zJUrQZC+Zk3GMffr6+WuXy/r0qWg6npPj/y6uqwTc2H77LGPHZPhuhPZDZMZhnzLkuG6so8dq8igPZnMrPbuTep6Zjc2Bq/bREJ+dfW0nzUSCfmRSLruwlx/5+TJSDpIP37clufN/Dqqq/O0c+fEanpLS3HbsUWjmUE6Fd4BlBJB+xJGgbDyF6Ygbl4KGVQtAaEKdiYHdhcvyqupSReoM2Mx+cuWZayuUttgdhmF6JYvD1arxwNJv7o6eI4WUYgu14ycGbcF3HijIufOSfb0it1+fb2ShiHz6lXF3vteObt2Kfbe92pqqexQf/ZMLT5XoTnP8XgqUDc1Njbz/ZxNm5S87jpFzp5Vsqpq+ut2cFDuxo1yNm3K+vM9Pal2bFU6ciSqwcHZJ2k3b3bSe9O3bCluOzYqvAMIM96SljAuostfqII4FFzYgp3Evn0aufde1f/ZnynS25teGfaWLdPIvffOf/WW2gbTjq1fU6PJO+WNsbGFH9sFZuRkzY655RY1f/jD2VPnh4YUOX9evmmq5lvfUvU//ZNqnn562kp72D57nB070mnwqTHI94NsitRzZdtyduwoyngKwfelWGxiNT31UOdkGBq95x4te/RRWVcuy6uqnpici4/Jr6vX6D33pJ8nx5FeftnWoUPBavrZs7O3Y1u+PDle5T2h3bvjWr68eAXkqPAOoJwQtC9lXESXvbAFcSissAU70fZ21X/5yzJHRyfGYhgyR0dV/+Uvy9m6dV6rt9Q2WMSxncce9UVl5GTJjsn2uWH29cka39+fXLNGflPTzCv5IfvscbZtU3LdOkVeey3zhkmr7sl168ouSyiZlEZGgiA9FstMe89FYtcuxd7zHtV+7WuKDFyemJyrq1PsPe/Ray17dPgbUR06VKVjx+w527Ft3Tqxmn7TTW7RLjFMU6qrk1as8FRd7VHhHUBZIWhf4riILm9hC+JQYGEKdrJU3E5XO5+l4nYoaxuEoV3iAo7tfPeoLyojJ8tzM+1zo78/KERmWXKvuUaKRGSMjEiWpeSaNbIuXZq2kh+2zx6/vj69gjyNaU7rPR9W8XgQqMdis6e95yJ69KhqnnlGilhy16xRzK/Rofh2/Xh0p9q/9Fad+4uVs/58S0vQjm3v3qAdW11dcVbTp1Z4r6kxtG6dZJp+Pho+AEBREbQjnBfRmJ8wBXEoirAEO1krbkvpr2etuB2i2gYLapfoeYp0dUkdHYpIcrZsyVuV+vke21z2qC80I2eu5yb1uWEfPapljz0mLxqV1dOTWUSvqkpeY2PWlfywfPakigB69fUyBwen3e7V1y+6CGChpNLeUyvqrpv/P1Dz10/q9GCLfhB9p340sFMvxLbI8WdOe49Gfe3YkUgH6tdeW7x2bNXVEynvUyu8V2h5AgBLBEE7AiG6iEZuwhLEoXjCEOxUQsXthbRLTAeyZ89KyaSWW5bcjRvz9lqb17HNcY/6QjJy0s/N0JC8mppgYsbzZB8/nvHcONu2BSv0iYTMgQEZyeS0HvNmIiG/tjb7Sn4IPnvM3l4ZAwMyR0ay3z44KC+ZDE1tENdVuiVbLGYUZNV4cNDQkSNRPf//jenIsT/VZW/VrPffsHZEu++S9uwpbjs2KrwDWCoI2oEKEIYgDkUWgmCnrC2gONvUIF/19fKHh2cN8hdkjmOb8x71XDNyUs9Nb6+MZFKRgYGMlXM5TsZz4zU1Bf3bk8kZsy40OhqMN4S8hoagLkNKqgDdpD3t5uiovIbsvcqLYWxsotp7PJ7/359MSidO2Dp8OKrDh6N65ZXZ27EtM4f1ltqXdGftUd0V/2fVfeLXlNizJ/8Dm4IK7wCWKt7uUBxh2DNa6QjiwqVAKdRhka647brysyxvpQK4sFbczjnwzRbkm+acFdgLYSF71HPJyLE7O2V3dQWBrOdNWzk3TFN2V1dmuvikQD3DTN8PkcjZsxNt3rKNfzx4j5w9K2fnzqKMyfMyq73nPe1d0pUr5niQXqXnn49qaGjm89aQpy3VZ3Rn3Qu6s/aYbq05qYjhBa0JFdPgPPq0L4RlZQbpWToNAsCSQNCOglvQnlGgjBU6hToMnG3b5G7aJPv4cRmJxLRCdJLkbtoU2omkXAPfRVVgz7OF7lGfb0aO2dMT7O32vBlXzs3BQZk9PcH9+/vl19TIGBkJ3uMta+JcSCbTkxtmf39Bno/Fsi5ezFxZT/2/lBHMp6rjF0qQkDBR7T3fae+JhPTyy8FK+qFDUb366uwRcHNz0I7t7aee0F29/6CGNdGc+7TnyjQz27BR4R0AAgTtKKiF7BkFylnRUqhLnb1imhp88EE13X+/rKtXg+AsFfCYppIrVmjwwQdDm12Qa+C7qArseVborhFmX18QsJtm9noFpinD84L7afy5rK2VW18vq78/sxBddbWSjY0yfD+0rSeT11wTTCwYhgzPywzeTTN4vL4f3C/PYjEpFjM1OmrkPe3d96Xz5630avqxY1GNjc2c8WBZvm691UkXkNu4MWjHFj16o5Y96sq4MiCvoUF+NCojkZA5OCi/ri6jT3uuplZ4n3o6AwACBO0onAXsGV1yKjyFeskpUgp1WLJXEvv2qf+xx1T/+OOyX3klWMqLRuXcfLOG77sv1BNyuQa+GUF+dbWMWEySZCh4T5tpdbsgFtg1ItrervrHH1fkxImJ86atbdqx8pYvl8YD82yLvYbnBXvZly+XlPlcujfcICMeTxcp9KuqZF26FOrWk7H3vlcNn/xksGXAtoMMASl47IYRTNY0Nir23vcu+m953sRq+uiooWRy0b8yw+iooRdeiKb3pl+4MPtl3tq1rvbuTWj37oR27Uqotnb6EU/s2qWhT3xCtU8+KevcOZlDQ/IjEbkbN2r0nnuU2LUrpzFWV2cWjyNIB4C5EbSjYMKUTpqh1CuU45ZCCvVSU4xzPmzZK2VbBDHHwDcdmL70koxkMr2aHBkvzuZblpzbbitaYJreoz4ehJvjQbiTJQiXgvNm+f33y7x6VZPzrq1//VfZp0+r77HHJlrErVwZFGcbGMie7m4Y8hoa5K1cOf25vHQpOC/r6mTE47IuXQp/68lIRMMf+5ga/viPZSQS6W+nX8GmqeGPfUwLrXrmOEq3ZBsby2/au+9LZ89GdOhQsJr+8su2XHfmKLi6eqId2549Ca1fP792bIldu5TYuVP2qVMyBgbkNzYGKfHz+OHJFd5ravzQngYAEGYE7SiYMKWTppR8hXJ8wiD6gx+o7q/+SkYiEaxWFSqFOp9CMtkRZgU/58OavVKmRRBzapdomorffbeqfvxjabytmRGJyE8mg73clqX43XeX5DVhjJ8HxkwBlOep4aGHZF6+HKSsT6k/YF6+rIaHHlLPt76VbhHnbNkSTFC4bhDITqoe70cicrZsyZigKPfWk87WrfIaG4N995OjasOQ19iY82RMLCaNjgZp75PmAfKiv9/Q889XpVfTe3tn73O2YYOjvXsT2rMnoVtvXUQ7NsOQs3nznHejwjsA5B9vpSiYhRZLKpRSr1BOnjCwrlwJLvxra4NeOyWqQj1fRZnsqIBJgUKf86HNXiljiX371PuWt6jm7/9e1ptvKnnNNUEa9NRIw/NU9dxzwQpyKpB1XRmGIb+2Vn4koqrnntPIf/kvRTlvp76fpbIE7FdemfZ+Znd0KHLyZLqQXLbCcpGTJ2V3dMjZvj1j5VzDw0quWBE8Tt+XGYvNuHJetlkX45NhGp+MMAcGFEkmlbQseY2Nsi5dmvM9OZX2nuqdns+0d9cN2rEdORLVoUNVeuWViHx/5hXu+npPu3cHq+m7dye0erWXv8FkQYV3ACg8gnYUTKGLJeWkxCuUGRfYNTWS78u3LBmxmKzz51NXPaEMvIox2VHyDIg8KXiBsBBmr5S7bOdezdNPZ22DFjlzRt7q1cGe9rEx2ZJcKf110V63Ob6f2ceOpfeYZy0sZ1kyXFf2sWNB0K7pK+eG6wZB7Vwr52WYdZExGZbaq2/b8hxHkmZ8T04kJvan5zvt/fLlzHZsw8OztGMzfLW1OdqzJ1hNv/lmp6Cr21R4B4DiI2hH4SywWFIhlHSFcsoFtjk8HHzfsoKLZceRurulG25Ip5+GJvAqwmRHqTMg8irLOa/xyZl8nPNhy14pd7mce9MmTKZEaMV83Rbr/axsV85zNN/JMONqb0bv9HymvcfjUkdHNB2ov/ba7JdnK1Ykx4P0uG6/PaHGxjz3h5vEMCaC9OpqKrwDQCkQtKOgwrLPsZQrlFMvsCfvJ5VpBntjx8ZkxGLya2tnD7yKnEJe8OAgrHu0FyHjnD97VhoakmFZeTnnQ5W9Uu5yPPdSEyZmX18QwE0pROc1NhZtwiTj/cz3g0r2yWQwEVhTM+39zNmxI5jscd2gqNwURjIZFLHbsSPLHyu/lfNczTYZlvTNYG+6sU4X4uuVuJCf9yHfl954wxrvmV6lF1+MKh6fORKORIJ2bHv3Bu3YbrzRLWjgTIV3AAgXgnYUXBhWaxa1QrnIQHnqhIFfXR1kHIyNTfRB9v3gonuWwKsUKeSFnuyo1D3a6XO+q0urJPUpT+38QpS9Uu5yPfecrVvlrVghu6srPflmmKZ8zwu2uYyOTivOVigzTSBohgkEZ9s2uZs2yT5+XEYiMa0QnSS5mzaV1Wssn6ZOhiV8W8Nug/oSEcW8Kll9V+Ru3KhE66ZF/Z2RkaAdW6rSe3f37AXkrrnG1Z49Ce3dG9f27U7Wdmz5QoV3AAg3gnYUR4lXaxa6QpmPQHnahIFhKLl6tSLnzwe/0zCC8biurO7urIFXqVLIC52OPeekQDQqs6dHVc8+K0nllZprmnK3bZPWrpV78eK0dOqFCkv2Srlb1ISU7weVx3x/4l8RlyKzTSCkg/BsEwimqcEHH1TT/ffLuno1aNuWGrNpKrlihQYffLB8Xlt55humLv/abyny4MMafdNXfFmz7NpaObFRWYNX5NfVafSee3I+xp4nnT4dSae8d3baSiZn/h01NZ527AhW0/fsSeiaa/LcxH0S25ZqqpJqeK1LdaM9MlYul3NNGb2/AsASQ9COpWEBK5T5CpSzTRj49fVy16+XdelSkNoaichIJrMHXiVMIS90OvZskwLG8LCs7m4Z8bjqvvQl1X7ta2VZnK4QwpC9Uu5ynZCyOztlXr2qZHOzrL6+oBaFxnt5W5aSy5fLvHo1HFkhWYLLxL596n/sMdU//rjsV14JqqhFo3JuvjlrX/dKl0xmVnv3btin6Mc/o9onn5R17pw0MiLDNOVu3KjRe+5RYteuef3e/n5Dhw8H7diOHImqr2/21fTWVidd6f3WWx1Fo/l4dNNNrfBed6gyin8CwFJB0I4lI6cVynwGyjNMGMiy5NfWymtsVOS3fkt927dnTaEuaQp5gdOxZ5oUMIaHZaUyEaqrlVy3TkYiUZ7F6QplCew1LqRcJ6TM3t6gH3ssFtzFttNt0AzPkzk4KL+2Nj91MebYkpOaQHDXrQsmEOLxYFnXNIPXS1NT1gmEpT7ZE4+nqr2bGhubfnti1y4ldu6Uffq0miUNSnJuumnWFXbXlY4ft9Mp76dOzd6OraEhsx3bypWFacc2W4X3iir+CXieIl1dUkeHIsrTVjQghAjasaTM96I134HybBMGI/fdpxXvf/+MKdSlbvNV0HTsbJMC0Wiwwu44UiSi5Nq16QJb5VqcLhSKXMSwpObzWHOckPKammTEYumibUpta/H9IHB3HGl0NHjPWIT5bMlJvSf4dXUZPdfT/7VtGaOj2d8TltBkj+8rXe19ZMTQ+Pb9RevuNtOr6UePRjUyMvPryDQn2rHt3RvX5s2ustQCXLTJFd6D4nEz3LECi39i6Uq/X549KyWTWm5ZcjduJGMEFYmgHUvPPC5aCxEozzRhYMxxBReGNl+FXKGbNinQ0xM8pupqJdeulV9fP3HnMi5OV0qlKGJYKrk81pwnpKYGyHN9fwFjn88KqNfcHKwuvfmm5HnT9rRH3nxTXkPDkmz957oTvdNHR3PrnR49enQiPd7z1GCaSl53nXrf/2E9b+3VoUNBpfdz52a/dFq1Kqndu4MgfdeuhBoaClNAbiEV3iu1+CeWnqnvl6qvlz88TMYIKhZBO5BFwQLlBaxyhabNVwFX6CZPClQ9+6zqvvQlJdetU7YlqVD1sS8DSykVdiGPdb4TUmZ/v/yamiBFPpHI6PxgpFLTa2pk9vcvbPA5rIA6t9wSbMp23WCrzaRJg9T7lpLJ4H5Z/k6lZVyMjQWBeiyWPe19PqJHj2rZo4/KGBlRclmDzkY26dm+rfrxi7fpyPM7FVf2CVxJsm1ft92WSPdN37AhWZC6hFVVSgfoC63wXurMLSAvsr1fjr8HkzGCSkXQDmQRmkBZWjptviZNCtR+7WtBYFSizIKKsZRSYRfzWOcxIeU1N8uvrVUyEsleiK6xUYpGF3xe5rICqvG/Kcua6L2emkAY79cuy5J9/HjG46qUjItU2vvISLCavui0d99X8q/+t/6/vt36QeRt+vGFnbrorp71R6691k0H6du3J5TlrWrRbDuzeFw+0urDkLkFLBYZI1iKCNqBbEIWKFdMm695rPKVdMIkrKuQCxxXaC9sCvA8F/qxZrRZk4I+7ZJ8SYbnyertXVSf9pxXQE0z6EAxvp0k3ae9ulrJlStlTtnTXu4ZF6m091S198V2UEy1Yzt0qErPP5fUy6f/UslZLolqjVHdftuobn9HVHv2JLRuXf7bsU2t8G7bef8T4ZqQBhaIjBEsRQTtwAzCFiiXe+Xnea/ylWjCJKyrkIsZV9EubHIIwgv1PBftsY73ZTe8oOp3OuRZ5PmY6wqob9uSbcvdsEHG2Fiw4h6JyK+uljE2lrlaWqYZF0Hau6nRUUPx+OJ/X2+vqSNHouPt2KrU3z/7Y7256lXdWfeC7qw5qp2j7Rq79/eU2LNn8QMZZ5qZ6e4znLr5FbIJaWAhyBjBUkTQDswidIFymVZ+znWVr9gTJgtehSzwyvxiV0eLcWGTSxBeyNXeRT3WeRxHu7NT1oULwfeTWVZZDUPWhQuLWsmfcQXU82ReuaLk+vWS52Xet6VFfk2N0gvPWVZLQ5txMYXnZaa9Z3uac+G6UmenrcOHq3ToUFSnT8++dN1kDuiOumO60z6ku2uOqsnqCyZB4nEZthRrbFzUeOZd4b3A0u+vjz+uyIkTMsdft05bm4bvuy/UGReARMYIliaCdmAuZRool1wqEOrpUf0XviBjaCho3zbPVb6iTZgscBWy4CvzeVgdLfSFTU5BeIFXexf6WOd7HM2eHpmDg+k09Iz87PGvzcFBmT09OY89+APZV0DNvj5Zly8HEe3581r+kY/IbW1V/O67FXnjjXmtloY5lTTolBcE6mNji097v3DB1JEjE+3YRkdnPpcsy9cttzjasyeud/74c7r13HcUSboyYgkZA758w5Afjco3Tbk33yxn06acxmIYE8XjcqnwXkzG+ICMsA0MmE2W90tZloxYjIwRVCyCdpS/sO5DXqgKeDwZgVAsJnNwMAgqRkZya+FWhAmThaxCFmN/cF5WRxeTCjvXeZhjEF7w1d4FPNZcjqPZ1xessI+nxWcYT5lP32+BpmWYXLkic2go2L++dq38pqb0+CJvvKGRe+9V1XPPzZmNErZU0lhMisVMjYwYSiQW97vGxqQXXwxasR0+HNUbb8x+WbN6dVJ79sS1Z09Cu3YltGxZcNxqq1bK/uKIlEwG9QosS77nyRgdlWFZQVr8PALbqUF6GN+6p573qdeJ/corZVHfAJCmvF+ePSsNDcmwrPKr9QPME0E7ylpY9yEvVCU8nmm9U21bGhyUEY/LOn9eyfXrMwL3UheMyXkVMpdgdRHlnvO1OrqQrQbzOQ9zDcLnfDzRqMyeHlU9+6wkLWiyKqfHmuOkg9fYmD1gn8zzgvstQjrDpKNDjQ88II2/ZlLPxeTxVT33nHq/9jXZx4+Ht7ijgqdtcu/0xaS9+770+utWOuW9oyOqRGLmYDoanWjHtndvXNdfn6Udm+8reviwvJoaGclk0BkgmZRhGMHWA8tS9PBhjf7Kr0w71wtR4X1Wi53ULdP6BkA26ffLri6tktQnydmyhXMXFYmgHWWr3KshTxVtb1djuT+eGS4IZZryDUNGMinr8mW5dXXp20pdMCbXVchcglX3ttuKNq7Z5LLVYL6vq1wnFWZ7PMbwsKzubhnxuOq+9CXVfu1rC56smu9jzXnSYZ4r6ItZaZ/4JaZkmsH4V62afpwmj29KW7eZfl+xi485jtJ70xeb9j40ZOjo0YnV9CtXZo+Mr7vOTa+mb9+emHPvuH3qlKxz5+StWDGx6izJldJfW+fOyT51Sv6WzRn70gtR4X0m+ZjULZf6BsC8mabcbduktWvlXryoRe+xAUKKoB3lqdJWCzxP9Y8/XvaPJ9sFoV9dHVz4jle0NuJxGbGY/NraUBSMyXUVslj7g/O+OjqfrQY5vK5ynVSY6fEYw8Oyzp8PgpDqaiXXrZORSCxusmoejzXX42gMDs7rT8/3fnPJ93lW6OKOvp9Z7X0xae/JpHTyZERHjgSr6SdO2EomZ15Nr631tGtXIt03fe3aOTIipjAGBoLnurEx3TZPkYh815UpT7XVrpaNdGt59IL861sX/sAWIV+T1GGubwAAmBlBO8pSxa0WHDsm6+zZsn88WS8IDUPJ1asVOX9ehusG33Oc0heMmZRmGvu5n1Pk3Ll5rUIWbX9wCVZHc3pd5TqpkO3xRKPBCrvjSJFIUKjQsooyWRW2fd5TFWJ8+S7umExOVHuPxRaX9n71qjneii1oxzYwMPuYNm1y0kH61q2OIou4mvEbG4PnOpGQqqKqTQyoMRGTbcRUVWcFLfWio+pb3SRn4X9m4fI4SR32875sVUAtGgDhRtCOslRxqwU9PRXxeGa6IPTr6+WuXy/r4kUZiURQmK6mZu5VvgJdCGVLM/VWrJCWLZN59eqsq5C5BKuLrcdc7NZ3Ob2uFjCpMO3x9PQE50p1tZJr185cpLCjYyJdfD7nwTzOm1wnHfyGhnk9h/O931wKtg99kcUdE4nMau8L5ThBO7ZDh6p05Mg82rE1edq9O669exO6/faEmptzW02fiWFI1rab1LBxuRq7jqg2OSwzEZfhj1ePr6qSb1lybrutZNlA+ZykLnV9g0pUCbVoAIQfQTvKUsWtFqxcWRGPZ7YLQr+uLh28D/3X/ypv5cpZg69CXQjNlGZqdXfLr6vT8Mc+puR1180cHBZ5BTxvq6PzCGRzfV0tZFJh8uOpevZZ1X3pS0quW5e1aJ9fVSXzyhU1PvBAekJhrvNg3udNjsfRW7FCikSC5t8ziUSC++VDCTItsvGTntyjxzXWPaSh6pWK3bh5XlXUs7lwwdLhw1EdOhTVCy9EFYvN3o5tyxYnXUDuppvcvD3UbBXe6356u+qPfHuienwkIj+ZlDEyIsOyFL/77pKtnOZ1kjok51WlqLTaOgDCi6AdZaniVgt27FBy40ZFyv3xzOOCcPDAgTkvYgp2ITSPNNPqb39bvU8+OetFa7FXwLOujuaQhTDfQHYhr6sFTSpMejy1X/uajEQi6ySB2dcXtDw7f17eqlVznge5nje5HEdv5Up5tbVBr/YZeLW18launPlx56jo59m4ZDJYTXeePSLjib+V8W9vyHAc1dq2qq67TqP33KPErl1z/p5YLLMd2/nzs19yrFkTtGPbuzehnTsTqq/PT0GpOSu8e56qnntOXl2dDNcN0uRdN6geX1srPxJR1XPPaeS//JeSBLP5nqQu1XlVcSqttg6AUDN8nzKLknTlyhU5Tkl2q82bYRhau3atLl68KA7bzL1mU8Fh3ma4C7xXLXVcr/7d36nxgQcK/3iKICNIdF35kcj8V8k9T8333BMEjpMr0EuS78vq7pbT1jZnYJ2N3dGh5f/5Pwer/tkufmMxGSMj6nviifmlEM9xbhTqNZtLFkKur5Oiva6k2Y+158k+eVKS5GzenHmss50Hizlv5vMad12tue02mQMD6d9pSPKl9N/yGht16aWXtKgN1tkUYb9sPD7Rlm1szFD06FEte/RRGSMj8hoa5EejE1tb6uo09IlPTAvcfV969dWIDh+O6vDhoB2b48zejm3HjoR27w5W06+7Lks7tgWIRDKD9LkOR8b7QnW1jLEx2ZIcKf11Tu8L+Vao98Qltg873+/Hef88wYJxfVy5Kv3Y2ratVatWzeu+rLSjbBVjtaCYe9UqafVjMSndhSwymPdaCIvcH7wQ6aB6aEheTY182w4uvo8fn76avICVoKKeh7NlZly5Inme3LVrZ295Nn4eLOq8mcdxtI8fD1L4x5dpfdMMVmJ9X0aqf7tlza8FW64KcJ75flBELhWoZ8xZ+75qn3xSxsiIkqtWZXSCSFZVybpyRbVPPqnEzp0aHDL1/PNRHT4crKb39Mzeju2GGybasd12W0JVUV/2qVMyLg3IH2uUs2lTzun3lqWMNmzRaG7PRcb7wnhvdtm2/PEnpeQ1RQqV0l6C969KUnG1dQCEGkE7ylq+qyFPVoq9aoV8PEW3wAvCQl4IlX0thFQQ3tsrI5lUZGAgiL7GC2bJcTKC8IUGssU8D2eaJEiuXy+dPy+/qSnrz009Dwp9AZ0qvueuXy9rvICePC8I3KurlVy5UuboaKgv0JPJid7psZghb4Zabum+5Q0N086bpCy9XLVHPzpxm577T7U6/voyed7MQXZdnafbb59ox7ZmzcQfjR49qtonn5R17lx6UjQ5j/R7w8gM0ufqwz6XjPeF6moZsVjwdxRMcOX0vlCg1etKmtStFGX/eQKgrBC0o/wVYrWglHvVlvjqRyEvhMq9FoLd2Sm7q0vm6KjkefIjkeAx+H6Qimmasru60kH4ogLZXM7DXAOVKfdP3HHHtEkCeZ6Wf+Qj8z4PsgZeyWS6hdxiL6BTv1+2LXfDhnQKtauJFOowXqDH4xpvyWZqbGx+P5PRt1zSZbdZPxrZoR+P7NBPRrZrwBuvkP9q9p/fvNlJ702/5Zbs7dimpt97jY0yEglFzp7VskcfzUi/NwypqiozSM9HGn1K+n3hpZdkJJPBhIzvK5Jj9fhCZ2ZV1KRuBSj3zxMA5YWgHciiaH3gPU+Rri6po0MRSc6WLYu/ACvzfYoFvRAq88rJZk9PUAjN84IAMvXcGEYQsDqOzMFBmT09koqzEpRroDLv+3teTufBTIGXcgy8ZpJxXra0ZKZQh+gCPZX2nlpRn63Y/UzitU36ib9X7Zfu1I/iu3UqvmHW+y9fnhxfSU9o9+64mprm2Hc4j/T7pqe+osTbt6umzkhXeC8Y01T87rtV9eMfL7h6fNEys5b4pG6olPnnCYDyQtAOZFGMvWrp4OXsWSmZ1HLLkrtx46JWZYqyB38xkwLz+dkCXwgtOM00BJMhZl9fELCbZtbJJN80ZXhecD8VfiUoHagMDsobX/X3k0nZXV2Lr+6e63mQJfCSZQUp7Plo25VlPLIsGbFYyS/QXXeid3osZmghtXrOn7d06FBUR45U6YUX7tTY2F0z3jciV9vrT2nnB9frjpbT2rTsgozl89+Pni39Pmo4qjHGVGuOqW5FvyLn31DfxY7iBKiLrR5PFfEli20LAIqFoB35FYLAJh8KvUI5NXhRfb384eFFrcoUY6VnMZMCufxsoS+Eck0zLWZBwtl4y5dL44F5trjM8DzJNIP7SYWdAEkFKpcvyxwbk5lMTtxmWfLGxjIDlUIXxcsWeCWTwURCntp2ZYzn7FlpaEiGZZXkAn1sTBodNTU6aigez/3nR0cNvfBCNF3p/cKF2S8Hrol0687qI7rL/4H2NL4i65d+WtHDh3Pejy4F6fe2O6aqaku11pDq4n2KeIn0VgZV2zIG3eyTogX4jElnVq1enVE9fvLWh9kyq4qWmYVQYtsCgGIgaEfehCWwyYeCrlBmC15Mc3GrMkVY6VnMpEBOFc/HFfxCaJ5ppqUoSDgTb+VKeQ0NMsf3HPuWNbGnfTxA9RoaMnqFF2oCxO7slP3iizJHRqbfmEzKHBmR/eKLi67unti3T71veYtq/v7vZb35ppLXXKPYe987ra1atsAr1W5wPoHXfKXPy64urZLUpzxta5mD52WmvU+eI5kP35fOno3o0KGg0vvLL9ty3ZlXxauqfO268Yp+avjbumvku7rBf02yI0ped50Se35aNc88M6/96CmTK7w3bIpqTfUFadgIzuUpWxm8xsask6KF+oxZbPV4qoiDbQsACo2gHXkRpsAmLwq4QlmIVZmCr/QsZlIgx4rnobLIx53vegXO1q1ytmwJ9m2nVpMn79uORORs2TJtMqkQEyDm5csyh4Zmv8/QkMzLl4P/nxzYjBfOm1wobqbAJlugVvP009MCtWyB1+RshLwGTqYpd9s2ae1auRcvakH56PPgOBO90xeS9j4wYOj556M6dChox9bbO3s7tg0bHO3Zk9DevQndemtCVVWSku9S9T+bGrt0Sck1azT29rer6YEH5mwH5+7aoepaI108LiOe3bFF3ooVsru6gmM1paCiNTo67Twu5GfMYjOrqCIOACg0gnYsXoXu5yvUCmUhVmUKvdKzmEmBXCuep0Tb21X/+OOKnDgxsarW1qbh++7LzwTQPNJsF/q4C1GvQFLGZJKGh5VcsSLdK9yMxWafTMrzSpD98stzB6u+L/vllxX/6Z8OAptIROalSzKHhmS4bvDzpjnj6mougVqlBE6xmBQbkRLHTsnpGZLfmNorPvfPuq504oStI0eCQP2VVyLy/Zl/sL7e0+7dQSu23bsTWr06swdctpZsNf/rf8m6eDGoLD/pNWHIV7UZV11TUnWv/UDuc/8/ubfvWtjk0NTXWoE/YxabWUUVcQBAoRG0Y9EqeT9fIVYoCxFcFDpgWcykQK4Vz6UgWFt+//0yr17NCAytf/1X2adPq++xxxYV/M43zXYhj3vWegUPPKChX/91edddt+BzaepkkuG6UikKH82359b4/YyBgSBYHx7OuM03DBmjo9NXV3MM1BYVOJWwFofnTaymj44asg4HgXL1uXOqmcde8cuXTR0+HBSQO3IkquHhmcdtGL5uvtnV3r1x7dmT0M03Z2/HJs3cks164w2ZIyPy6upUZSRUawbF42qMMVkjQ7IuX5IRi8n7/J/Ir6vL+rqyOztlXr0qd906Wf39menxNTVKNjbKvHp10Vsr5m2xRQapIg4AKDCCdixaxe/ny/MKZSFWZQq90rOYSYGMiudSEKWMX6Bnq3guz1PDQw8FadVTU2ddV+bly2p46CH1fOtbC7oILujq7Sz1CjzXVeT8eTV++tPyGxoWtR83DIWPktdcM+/7pZ/zWCzzxvFjKmnxWQ4LDJxKUYvDcZTemz42NpH2Pp/e5UNbd6mjI1VArkqvvTb7x3hzc1J79war6bffnlBj4zxy7GdoyWbXWKpbGVHDyAVV974pv3nDxKTM8LCs8+eD2gqmGdRVMM2sr6v0Z8aaNXKXL59Wf0C+H2y/GP/MKMZnzGKLDFJFHABQSATtWLRKSUstmkK0jirwSs9iJgXSFc+TyWAPs+9nBO2Sgmrj4xXP7Y4ORU6eTK/EZ1uZj5w8KbujQ8727bk9kAKv3s4UaBrDw7LefDNoP+b7SjY0zBjQzFuJCx+5ra3pyZQZGYbcG2/Uss9/XkZ//5z3tS5cSAfhCwnUcg2cilmLIxabqPaeSGR7QNkDZa+qWq81btCPL23WDz+1TkfiqxWPz5zlEIn42rYtkd6bfuON7ryTIlJSLdnMxjrVWSPp1fSIkZRsX5FqV0Y8LjcWk19bK/m+rMuXg9e4gv3tfm2tZBhZX1dTPzOm1h8wxsYyPjOK9Rmz2CKDYZhMAwBUJoJ2LNqCArqFpqNWSEu5QrSOKuhKzyImBbyVK+XV1AQp8pOlgndJXn19uuK5fexYetUtay9yy5LhurKPHcs5aC/06m3WQHNSQONHIjKSSRmeJ6+urqxrPpgDA/Jra2VMrh4/JYj3a2uDSZgzZ4Igrr8/877pOwY/M3mbxEIDtXkHTgXeJ51Ke0/1Tp+r2vvk3uUjfq0ODW/Tj0Z26kejO/Wmsya4Uyz7z65b56ZX03fscFRbu7DCeJYl1dT4atIltXivyqpvnv7YDUPJNWsUeeMNmVeuyFu9OpiMGhtL/5Lk6tUZk21TX1e5fmYUdc/4YosMUkUcAFAABO1YvBwDm4Wmo1ZSSzmpMK2jCrnSs9BJAeeWW4K/PzVIm/y1aQb3K7CFVDDP5XFnDTRHR9MBpqSJiYfRUSmZlFddrcjp02VX88Frbg6KkdXUyOrrm8iikILAbflyKRqVpOkTGdmWfg1DmrRNYlGB2jwCp0Lsk04kJgL1eHz+1d49Tzp13NeLQx/QD4d+Si/G2uTO8vFcXe1r584gSN+zJ6H163Ps/zbONCfasE2u8G6vXyY7KvnxeNAub8rrRLYtr6lJ7nXXybp0KZi48bxgwmPNGvn19Rl/Z9rrKtdJQPaMAwCWuFAG7d/5znf0D//wD+rv79f111+v/+v/+r/U2tqa9b7PPvus/uIv/iLje7Zt68knnyzGUDFuvoHNQtNRK66lXEohWkcVcKVnIZMC9vHjwRKeFbSb8k1ThiRfkuGNV6u2LNnHjwercDt2BIGv6wZ9yKcwkkn5ti1nx46cx58Kqs2+vuz9oRsa5HuerNdekz3psc33cWcNNFNV0g0jnUFgdXdn/G1Jiv7wh2UVtE9+rM7NN8vs6ZGRSMiPRuWtXCnr8mU5ra0Tx9PzZk6nHw/YJ2+TKHSgtqh90uPt/PyXOpSImxq8YYtGx6zsae8z6O83dORI1XgRuah6e++WdPeM97/JflX77EPa+Rtb1Pbza1LzIUGF/pOnZAwMTKo0nz0f3jCC/uupIH3qXEhK+ti+9FKQGTLldeJblpzbblPv174m+/hx2UePatljjwXHaJ5ZEenPjPEOEeb4RKwzQ4eIJblnvEIyywAAixe6oP3HP/6xvvrVr+ojH/mIbrrpJn3zm9/UH//xH+tP//RP1djYmPVnampq9NhjjxV5pJhqzsBmoemoFdpSrigKcdGX46SA2dsbTE6sXy+rpycdABiGEfR2XrlS5uhoOjhytm2Tu2mT7OPHgyBwSiE6SXI3bVpQgOts3Tpzf+jRUVnDw5Jta9ljj03P5JjP456hXoEUrDanAnc5zsTfTiZlJJOq+8pX5Nx22/yDj1Jf0I8/1uX33y/71KnMKv+9vfJWrNDw/v3B8Wxtld3REYwvtSKfihZT/2+aQfG18W0SUmEDtYWm31s/+JGMP3tCYyfP66xbpYRlSzfeKN17r5SlunuK60rHj9s6dCio9H7y5Ozt2BrMId1Rdyz4V/OC1vWflLtxo/rf+/Z0+7dsLdmmVpqvrtakIN2f3/5201T87rtV9eMfS+PbOmRZQRr8yIgMy1L87ruDrgXj6e7V3/vegtPXjVQxuzkGt5T2jFdaZhkAYHFCF7T/4z/+o975znfq7W9/uyTpIx/5iF544QX9y7/8i37pl34p688YhqGmpqbiDRIzmyWwWWg6aiW3lCukolz0zSNwTAVHsm25GzZMqxQ9teiUTFODDz6opvvvl3X1alDcKrVKKynZ1KTB//7fF3+h7vvB753y/75pKrlqlYyhIdkvvaSmj39c/Y8+qsRP/dS8fu20egXjGQPpPe3jxyJdddv35dfUyHCceU8+hemC3pcm6hNMCsbTIXxqIuOBB6SRkcz7pn5HVZX8qqrMlm/jChWo5ZJ+n0gE1d7d555X9FN/KnO8qJ6hmCzFZL30kiL/9m8a/P3fz2jLdumSqcOHq3ToUFQvvDB7OzbT9HXLtf26u+/vtc//oW5Z/qbMqoiMREJm/6D8ujqN3nPPRGbGDJXma199RSv+708p8Uf/Tdbb36IsySpz8zxVPfecvLo6Ga4rI5EIJlsMQ35trfxIRFXPPaeR//JfguOwgKyIqdlTqfvbr7wye/bUEtgzXrGZZQCABQtV0O66rl599dWM4Nw0Td166606derUjD83Njamj370o/J9Xxs2bNAHP/hBXXvttVnv6ziOHMdJf20YhmrGV1nmmuUvtfmuRpSU5yky6eLanXRxbfb1yXDdudNR+/oyHuNCf262sYRJIY5rtL1djbNc9A088siiL/qi7e2qf/xxWWfPTqzybdw4LbXVvfVWJVtbFTlxQsmWlsxK0ePBkdvWJvfWW9PPgXPXXRr4sz9T/Z//uSIdHTKHh4Mgz7JkGIaW/T//j4YtK+fHEOnqSp8PZl9fsPo9mWnKcF1FXn89uM33paEhLf/N31T/F7+oxF13zevvOHfdpb4775Td2amVkoZefFF1f/Znsrq7J1L+PW+iPdaaNfItS5GzZ2V3dQVbJmZQjGM7L56nZQcPyojF0pMvkoLAbnwf9LKDB9V7553B8fzc57Tsj/4o2C4xfix925bX0BBMXNTXa+S++2RkizItS+5tt6W/zMsrxbI0ct99anzggWmBptHXr5G6lbrw4U9o4M2oHEeS76vpsS/JvHpVkoKJGMOQ7/sykkmZV6/KOviEjvzGW3VoPFD/t3+b/SN25cpUO7aEbr89oYYGX/bRNar76zGZ50ZkDAeTW8mNGzVy771ydu0KHrvvq2680ryxermWmXHVmsOqrR5TpMGV1d0t94nH1PvOvZKR+3tepKtLkbNn5a1eHUwoxWITE23jX089V5277tLAI4+k3xPMgYFgMmk83d3Zt2/iuKXOnVmyp1LnTines0v6WRvy56aclcU1FBaEY1u5OLYTQhW0Dw4OyvO8aavmTU1NunDhQtafWbdunX7zN39T119/vUZHR/XMM8/owQcf1Be+8AWtWLFi2v2/8Y1v6Omnn05/vWHDBj3yyCNatWpVXh9LIbW0tJR6CNl9//vSww9LJ08GFZmiUWnzZunAAekd75A2bZKqq2V5npQtAB8dlaqrtWLTJmnt2onvL+Tn5hpLqXiedOyY1NMjrVwp7diRvvDK23H1POnLXw6el2uvlZV6o6uqkpYtk958Uyu+/GXpl3954Rd93/++9N//uzQ0JK1YEfzueFyRkydV9d//u/TFL2Y+z5/6lPTrvy7r8mWpuTnI2R0bk3p7paYmWZ/6lNam+n6nnqOGBulXfiU4hq4rNTZKq1fLSiRkzfR35tLRETwvo6PBxXA0Gvy9VK/w8dV2Ix6XUqvhyaSswUGt+J3fkb7yldz+3vhjanz3u4OvP/lJGb4f/D3TlGpqpLVrFVm2LPjbQ0NaJWWe/5NNPrbr18uKxYLnMRKR1q+XLlxY/LGdr6NHpVdeCcbi+8EYUlsNxsaCFm6vvKK13d1B2vj73x+M60tfCh7DhQsyJJmTXpsriv3afP/7g/P34YelV05rZMDXcKRJo5vvkPeRX5f51rdqeeq+HR3Sa6+lf9RwXfny9Zo2ql379EPdpcNn9ij+O9Uz/jnblm6/XbrrLmnfPmnTJkuGUSNpUnr+z/2c9O53S8ePS3190vLl0i23KDp+PC1Lqj3zkmq7X1JtiyW7xgmOQdwNjkFtrbRypazXX5947nPV0RGsrNfXB+eRbWfeblnZz9XUMZ70HhfZsUNVU8/Fo0el11+XVq2Sld6cP8lix58nJfmsLZPnppyF9hoKi8axrVwc25AF7QuxadMmbdq0KePrj3/84/qnf/onfeADH5h2//e97336xV/8xfTXqZmbK1euyE1duIeUYRhqaWlRd3e3/HwULMujaHu7Gh94YCLVsaEhSI188UX5v/ZrwerfHXeo+YYb0iuuU9NRrZ4euW1t6m1pkS5enLitpSWnn5tzLP/jf8hrair6CvxMK9MjH/uYmv/9v8/bcY10dGj58ePyGxrkO860ys9GQ4OM48fV993vSqaZ+/PgeWr+zGcUGRiYOB7JZBAwrF4drPJ95jPq3bx54ve1tSn62c9OPP6rV4PHv3lzsDLf1pY+dun7jIzIHBgIAkLTlBIJ+YODSq5eLX+mvzPXc+N5WjE6GgRck1PUp9zPt6yJ88w0ZRiGvMH/P3tvHidXVeb/f85dau/q6j1LZyErJCFk0bCkQxBUojKiw/gaRXScURBFYNQZwRF3VBgdhkhQdHRGMbj99IuorIqydAIE2Zqks0BC6HSW3qu7a697z/n9ce69dW9tfavT3aluzvv1yitJ961b525V53Oe5/k8IxW/n/2ZldesQV1jI0+RlyQw85xRCpbJ8LIBWcYQwE0Ji43fuLZQFEivvFJopFdbC3R2YuhPfyobrZ8IPPv2oX5oiI/fdi5BCGCUAWBoCIP79iEzZ07uhZdeCvzd3xXPgilx3JOVNZNOA4lZKxC/+W7oL79imblppplbb6+1rW/HDtRksxhFEE/jXDyJTdiBNhxFa9n3aG3VsGEDj6avW5eBvXy+r6/MC5ubgeZm7vCe6HM4vHsGOxHK9IEmA6BH+gvuA72xEVIqheiBA85z7xIFQJ0sg8Vixev9k8ny9+qcOfwPAPT0FPzac+AAIqkU9HAYyM92AQBZhlxq/FOQQXUqv2tP6twIylLNcyjBySGu7cxlpl9bRVFcB46rSrSHw2FIkoSovZcvgGg06rpmXVEUnHbaaThx4kTR36uqCjU/amAwXW4GxtipGWup+mVKEdy2rWw6X3DbNqTPPRejY9Q9jl5zDVi+wzQh7l+n6+XH0tWFyNVX8/7SU1gLXCqlWdm7F+HPfQ6orwc744wJua7SwAA3cFNVKMePF53Qk3gctZ/7nOWgXcl5UF9+GfIYHgPyq69CefllR+1peuNGpM89t/g9xJjzHHm93BTOPB+Uctf5RAJydzf01taS71OWvPvK+rvIPed4jSSB1tZW/n7WLhiyK1e6cuTOrlxZsouANDAAEotx0WQahJnR7WQSUiYDFghAGhg4+XtpDL8CaXDQui7F7gMmSSCUQhocLBwLIcieeWb+SSo6jIms32cMSCaJ1ZbNsU5rW/y1j4dS4JVXFDz/zFo8i+14AWuhl/nqDKgZrN1ArbT3OXOc7djGuiyE5Nqw+XyFDu+MgTvs6zqUo0f5Nci7D5SjR3mde13duO6D7MqV0BYvHrvev8y9Wg5aV8cXr8YyAcwb/1R7OUz4d60bD5BxnhuBe07ZHEow6YhrO3MR17bKRLuiKFi0aBF2796NDRs2AAAopdi9eze2bNniah+UUnR1dWHtONpBCUpTbrLEwmHXRnHjdYN2+7pypnUkHodk9MbWwmHQ+np35j4n69Ltwv0et9wC/N//lWzVVAm0vp5Ho0pN6I8c4RPt7m7QpiZnTfQNN2D04x8HnT+/5LGeVKusUiZS9nPU0gLl8OGcSZyB6RxPNA1yby+0BQtKv08JpGiUZxvE4/w+NiPq+aLd9m+i62A+H1hNDaS+vorez/nmLh25y9xbNBLJCfa86DZTVR7dTiT4/X8SuBFHtK6OZyFQimJfo4RSvthhtnBzHIi7Z2rchly2/WdqGzB82iokUjKSSZJ/WxVlaIi3YzOd3qNRCUBhuZXJ6diLTXgSbXgCSz5zMfR3vX3sN7FRqcN7dsUKnt2iaWBeb+F9kE4Dus63Gw+T3G6vEhNAk+luzuZ2wWE850YgEAgEM5+qEu0AcMkll+DOO+/EokWLsGTJEjzwwANIp9O44IILAADbtm1DfX09Lr/8cgDAb37zGyxduhSzZs1CPB7H73//e/T19eGiiy46hUcxsxhrshT/wAfGFnHRKNTnnrMm6WZ/30qEsBsX6ZKCkjFeT224kBNZBpOkMdvGTURkx437vbx/P5Tduwujj+Og7ITeiOCAEOitrdaxMr8fVNOgdHej9itf4an1JY51vK2yymE/RySd5in9pdQVpVy4jow438etk30gAC0UghyNFvRKzx0EcRjF6c3NVvu5So4rf9wVOXKXwp4h4ObnFeJWHNHGRtBwmPe7ty+AGMZsIKSghZu5f1fP1DhbPXra26F+9wdI7T2CWNaDpBKEvmgxsldcAVqiBljTgN27VTz7rAfPPOPFgQPFs7FMIhjCRrRjE9pxHtrRDCPPXVYQbXgf9LKv5lYKpkj3+1nF2lft7OQLPrLMSz3yz73xO7Wzc9xO65PaF73SRYFp3vazogWHSV4wEQgEAsH0pOpE+3nnnYeRkRH8+te/RjQaxcKFC/Ef//EfVnp8f3+/w0EwFovhBz/4AaLRKILBIBYtWoSbb74Zra3l6wwFLnExWfI/+GDZdD5paAhkeBg1W7fy19om6WljMcY1xSK1NrEmDQwUHQtJJvnPJAkE4FFO65fF28ZNVGTHTWQao6Pjj+DmUXZCb+YDE+I4RyQWg2xE5gljvJ5Skooe62REguznSIrFuJgth65DGh5Gds0aZFetGlcUS1u4kC8QGK7YyGahvP567jxJEu8j39wMFgxCPnHipCJc1sJEc7Pltp7f+m6s1oWOTIFMhqemG5iRbeb385Zk46ECcZRdtQrZlSt5ur+5CGFP91eUghZu1jM1Ogrq9/NsAUqhdnYW3GeVtHrMnLkaySRB5i+7oH7hv0CHRgEmA0yDTEYgv/hiQUu248d5O7Zduzx47jkPEony7dhWnjaCC47+Euen/oQV2A0ZlH+WWGOSQGtqwIpkFqiqM5KunOQ3rzQ4CEgStNZWyP15Ne0+H69pTyRO+jNlMvuiV7IoMK3bfo5jwWFSF0wEAoFAMC2pOtEOAFu2bCmZDv+Vr3zF8f+PfOQj+MhHPjL5g3qD4mayJPX0QJ81C0p3d4GII6OjkI8fB2QZur0XbznhW0E6ejGxRpJJkEQC+vz5ubHoOhdjRk9s5nM6PBekdE9gZMdNZBoez/gjuHkUndAbGQZMkrh7OUz3a1hZCFYfcV0HoRQ0GCx+rCcTCSpxbR3nyKUhJFNVxK65Bp6dO8cXxerp4dsHg3zs8ThoSws3f9I00NpasJoakEyGt2obx3HlXxdr8YYQZ+s7jFFWYL6NkSmgKwrk/JZ1sgy9tvak7qVKxZF5LhGLQW9osFqgSclk6Wjp4CCIrkMxTQYNkY+8PvVjLXZlPQGkhih6Xk0iGpLBKEPdf/0ItD8K2M+s8c90fwzPf2cX/nLuZuza5UVXV/mvv6YmHRs2ZHD22WmsX59BTYii7so/Qn3lFYAVzwLRm5uRXbYMsuyMpKtyhSU2Y9xP5vMCVYV22mlFF4BOKivEziT2RXe7KHBSJTmnmPEuOEzmgolAIBAIph9VKdoF1YPbyVJqyxYEf/lLp4hLpaB0dwMAtNZWS7CWE76VpKOXioSTeBxSPA50dfGoptfL81+NSKTe3Oyo8bZSyWXZmuROZGTHTWQaa9ZwF+QJIH9Cb2YgEE3j0Vgz7TyT4YdjZiGY4T8jjb7csY4nElT22p53nnWOaLE+3UVI/t3f8Y4El18OMjTEr5UpAk8iigUg97u+vpM6rqytr/tElBVkV60CbWiAumdPrn4ZXJcSSiEPDhZEtyuhUnGUfy6JxtuOlYqWqnv2cF+JIl4LRJKg7tlj3WfFzleKehCjAcSpH5kkBZFqEPM1gDFA3b8fysGDMFU6A3AQi/Ekzkc72vAsNiBz1Av8Jv+oOB4Pw+rVGcNALo2FC3Xno29fYSESXwAzFilkqsFPElCVQQTnavD6x19i42Z7x2fKrFnOBaDpVvfsYlFgMkpypopJ8QARCAQCwRsOIdoFZXE7Wcqcfz6ya9Y4hZDhLK23tIDV1OS9sFAMVpSOXi4SPn8+0NXFU8BjMT4WWQYNhwHGwIJBng7e25tLK6UUNBwGGR4GMMGRHReRadx4o+WifrLYJ/Q0FILU15dzGgc3VgPABanXy9/TELtE07jpmi0TodSxVhIJcnNtzXMk2dpslYQQ6EuWILB9OzzPPMNdymOxnEN+czNYKFQ+inXOOfDfdx/ko0ehz52L5KWXwsxbnqjjGr71Vt67GpNkMCVJuW4LRjbFyTAeceQ6WtrfD2lkpGiLONNETxoZgdTfD4Cfr8zipcjufR0j8hzEWQCa+ZXFGOSRPmiLFyNruL4ru3djVPfjKZyLdkOoH0f5lljz52t485t5NH3NmgzyEnAcqAcOQIpGoUdqoQwPI6CPIIgE/EjAK2mg9XVA9DCkV8ZfYuN6+zdY3fOUmrNRCmXPHqCjAwq4k/7JnMfpvOAgEAgEgupBiHZBWYpOlswItabxydKZZ1qTdPvkXX7tNdRs3QpmREAd/cJ9Pss1W33uOWRXrHC4h5NUCiQe5ym/LS2Qe3ocEdMxI+HNzSDxOEZvvBG0oYE7xQ8PI/L5z0Pu6spF+8x0cUMwRz7/eURvuWXCJ1rlorvxT30KDRdeWLpHdaWYE/obbuCZDmZUE4ZgVxR+PbJZyCdO8EUPgKday3IuE8HNsbqJBLktNbjnHkRvuQU13/oWPC++WHp/hIB5PIAkoeb2261opHkNSTJptYVjgUDRBQezH7yyd6/1ev//9//xnvGGKDrp4zp+HDXf+hYQCkGRJGRXrjxpoaXu3g1pYADanDkFRnrM74deWwtpYGDctb2O593IlrE/syXFkYvzJZk93cdoEUf7oxgeJkgkFBy57HOo+c5/gfTEQcMewCOBZDKQRkbAgkHEPvBB7Nuv4plnvPjbfX+H3fh42XZsQcRwjvQM2oLP45yF3aj7l4utGvex8CcG0ZA8imBqEAE2CigSJAAUAKGMjykQGH+JTYXbv6HqnqdokcLKcjh4ENB11MkytMWLT+p8Cjd4gUAgEEwEQrQLypM3WWIeDyRTLBjO2lI0Cs/OnQViRzWErzQ0xB2mTYGR11qrZutWBH77W8hdXWA+H5TDhwt6WNPaWkfE1G0knDY0OMzuot/8Juo+8QnLtZvAmBTbzMZCd96JwZ/9bMInWqUiksRlOnglZNraMPrxj6P2K18Bsbl5m8ZqACAbPdylwUFr8UKbM4e38RodHVuouaQiQ7G2Ngzcdx+azz0X8rFjxXfIGPTZs+F78EEgneame2bbNlvUVu7t5anDeQsOnvZ21F1/PaSBAce9KD/9NNRXXsHQ1q2uJuhjtRckiQTUjg7gQx9Cnc9nTf5PRmhZ931LC7S6uoJaZjAGqbf3pNrSxa65BnXXXw91//6C3vW0oWHc4qhci7gU8yKu1yAu1aAnOw+ZfmP/69Zj9LOfReCee/hi2+goekkz2mvfiyfq341nvjYbw8PmWEJF33cF9qANT2ITnsAavAipvg4sGITUPQL2X7sx+tnPFhXu+Q7v3lEPGtJHQXQNIMj5ZBj3HdE0R7u9SktsxlOS80aqe57sRYr8LAeEQmCx2Mm3lKvmrIiTbWcqEAgEgilDiHbBmJiTpfDNN3Nnckq5oAsEoNfVQT5xouikxlF/C3AXc1NAGjCfj7c8O3QI0uioJR7z612lTMYRxRpvJJxFIrzlVzjM276ZYseYJFsT487OyZloTWGNIp0/Hywchh4Oc6GUd6zaokWQjx1D/KqrQCMR1Nx+O+/tPoFCDRhHqYEkgTY08P71xdq+EQLIMhc4TU28Vj+Vyjmp21rbSf39yK5enVtwoBThm2/mKfhm7b7NWV/q7UX45pvR/8ADYx5vqeMisRjk7u7cfR6JgOU58Q/ec8+4Jsv5932+md1EGZAxgJ97W+kECCnaj90t9hZxLKMhLtcgRmoQZwFQHQABaKjQfT2+ej2eIefgbw/H8MxLtThwrBYYAHC4+PvUYRBteBJt2IGNeBKNGLD9lkALBPjildcLua8PgXvuQWbdOqge4hDpRdfSzDIEa3fEuRBpe3Yc90d+ppHfX3Dfj7sk5w1U9zxpixTFshxctAStZNzVlhUxEe1MBQKBQDB1CNEucEXmvPN4jWVNDRe+qmoJQJ2x8pMaY1JLSggw5vOBNjbyaDylPPUZyDmeK0pBFGu8KYfS4CBvleX3g9lEiTWhliReWzs4iPQFF1TdRKsSLEM6SQINBgt+b56H9AUX8J7neefQSs+t+I2d0Rtq3C9uF1jMFHC9vh7y0FBh+zdCoHR1gXo8oA0N0JuboXR3F/QKh64DXq9jwUHt6ICyf78VkS9WV63s3w+1owPZNWvKH2axhSO7E78s89ILjwdMVQtbpplCyzxf/f2QhoZA6+pAGxuLipFJT7U1xcvoqFOgGiUkZHS0+HPuImKXPH0V+pafjczLh5DUPEBWy2XTeFQwWYa2ZAmyy5bh2DEZu3Z5rHZsyaQEoPhChCwzrFyZxYYNabz1kS9iddfDkEp0Smder+XXIBOKYERGsPs5NIy8BKw7s+ypkQYHrc4LuR06/08YK1hYLMg0smUP2e97Uft86piKlnLVlBUxUe1MBQKBQDB1CNEucIW6ezeUgwdBm5oKJ5QlJjXq7t08xVmSSvbdJpkMr5u111CbbaxsET7zfSzGmXIod3WBjIxAMftY50WVAYBJEuSuLgDVNdGqFNcCb8UK1H/oQyC6juzy5Y7e5czrLfATKEfR6M3ixVbk3I3QlAYHeXp5MlncmM9wwJd0HWxoCLShgbe3sxsLgreEG/nXf3U6mL/wQq4ve7G6aqO3vfrCC2OK9mLn1+7ET3Sdt98KBHgbuRLmi6E77+Su6oZJGyQJNBy2auAdk2f7fX/8OKjfb9VDS8kkWE3NSWVFqLt3Q33xRd59och5l+JxqC++6HjOy0XsRt7UhkRCQiJBkE7L8LzvGtR0/RdIPAYaVCBRCipJSGkKdskb8Vf/VXjqg43o7i7/1dTSomPDhjQ2bMhg/foMQiF+zT0r3w72jWeBaNS6T8xlJ0mW4GkJwq8OwU9S8EpZQKWQ40OIRvtAOzrKG+kNDXEjSyPFPx8zS0gaGgJQxOk/L3tITiQcTv9TVvs8jVOiJys6PGUt5aohK2IC25kKBAKBYOoQol3givFMaiy3aPA0eKLrlnixxJiuc5FuivMiNe+WCFMU3h7NoNKUQ097O0I/+AGPlpmTb1tPcKaqIJSCMIbQD34AbdEi96Zk1YjLhQ21szMXZTJSQu1S2W2UqWT0Zt8+HgGXZVcLLDQS4eI3v1+7KWLM+0OSIPf2gtbVgYVC0IJBXuOdzYIMDyO7ahWSV1wxcecznyLnF9ksN1g0SkhoczPs0177c2Kdr8FBpzEipZCGh6G+9FLRqFemrQ3xK65A6I47oBw7Zi1u0ZoaxK+44uTES28vpNHR3A/ysy8ASKOjlsN//jXXPD4kUxISewYQ++z3EP1syFEvnlm/Hsl3vxv+u3+Gg9EWtLM2tGMT/ob1yMIDPFl8XB4Pw5o1GWzYwNuxLVigF6y5mPsf/cIXENi+Heqhg/CnowglB+H36lDn1IOFEo7tTW+Omttug9TTU1YIUiNtv2jGkPlzWba2G5P8A5iC2udiBozaGWfkDBirmMmMDr+RshymIqtAIBAIBBOPEO0CV4xnUpPvFm32NeYvyAl3ZqQxF0S9bSnaIIQvGhjp8SZjte7KHYARXYjHobW2WunUjmPQNN7bfO5cSLHYjIg2uFnY8D722MlHmVxEb/RZs7ih4L59kAzBkC0lGPIFepHJpV5XB3lwEHJ3N88AMcZP4nGwSASxT32qML187Vp+H2saX0jIg+g6H9fataWP1UZBn/Jkkg/b44E+ezZv52ffv/mcRCKoufVWkNFRvphla4PGwLNNiKYBRe5DT3s7gtu3A4oCbe5cq1e4lEwiuH07b5U2TvGivvxy2XNuLqKpL7+M9IUXInTnndBGUxhpXooYDSCp+cAUAjQwR704CMHICMGLv+7G879Zhh3JB9DDmsuOZcECDWefzaPpZ52VQYnb04HPB/jfug7+d52FmldfhjzQj5rbbuPdBPJLRAzTPpJOQz5yhJcllBGCtL6+bNYQAJ4lkVfmUYnT/2TWPlsGjP39Dq8C+amnKjJgPCVMcnT4jeTwPmVZBYJTyzTOqBEIBMURol3gikpSrVUjzZREo1xsm27RZqsnez9pw1TMLthNF2zCGK+zNgR7sdBasXRJ/29+UzDBtUcXmN8Prbk5F6W0oTc1gdXUgCrK5EQbin2RToJ7vJ2xUvwnIsrkJnojHzsGKIq1cEOKXE8APKro93PRmh/VNBdyJAkIBECzWeitrZAGB10JnOzq1dCWLYPa2clr+vOM6ABAW7asomvuOL/9OZHIAgGQBI/sEsDhxA+Any+/H4oxbvt5Y7IMkslAb2hw3od54oWkUnzRS5ahh8MVlTIUxWWf96SmYmTHfiT2pZAOrgLT8xqcE4JsTQSdB0N49DspPHVoDvbuVUFpaaEeQgxnRzqx7mPLseHsDFpaike07Xi9OYd3n4/ZDplAO2s1NPAFlKLR66Ehfm97vdBnz3YlBBn4tSyF/dNkvE7/FZfkuJmcmwaMPT1FS06knh7XBoyngkmPDhfLmpFlboJ6qh3eJ5g3UlbBjKICES5MBgWCmYkQ7QJ3uEjdTG/ejPoPfcj6ogBgiXTTJIwpCv+dLcUZus6j8oaoN6OgzOy7rWlW2rY9Pb6SdMmC6ILHw12cJYkLKkJ4xNMwwZuMaEOpL9L4pz4FvO99E/Y+AIp+wZeazE5ElGms6A2yWd4qUNN4ez1b6nyxiCYLBKB5PFD6+pwO5pLEBS0AxhgQCGD41lt560E3AkeSMHLTTYhcfz3kgQF+zW371hsaMHLTTcVfX27SZCuhYB4Pb5m2b581dsVYfKKNjYhdcw0/F2Z/eXtGiYm5kGA8D+Z9aIoX5vG4ao1YKfq8eVaNvH1clBHEEUAcQcRJDYZCZ4B2JxDOErBaj/X6Pq0OO+JrsSO+Dk/F12CYhoE/lH6/lb5XcF7gebQFn8dq0gFPKoaR5d9AtmW5c0PGoB44AE9sCN6WGqhrl8MfJK7Wu0pFr7V587ihYX29KyFY1Iguj2JGdONy+ndZkuN2cq52dEDZu7e4RwTA79G9e4sbMFZBxG4qosOO++TgQWB0FESWp43xqFveSFkFM4VKRLgwGRQIZi5CtAtcY01qjJpIe4pz+oILENy+3flFkUo5zZvMtFJDjFuO1JoGbdEiPoH2+wtTSX0+6LW1IIzlJrgVpksWRBdsvb2ZKVJMsyhMfLSh3Bdp7Q03AA0NgBGBLYuLCXTFq+wTUEtbNnpjOKqDMZ7JYPy+1LWyJpWdnTxanUpxsW5f1PF6ISWTfHK5enVFIiLT1obo1q0IbdvGhXUmA3g8yJ5+esna3orPqVErXVDykU47zpeVdZIv3M17nzHAdh/aTfqIro/ZGtF5kVw4vF96KcJf+hKk4WFkmYIYCyGOAJIIgBkxZhqqQeqii6C++irSSgDPD69Au3Y2dsTX4UD6tLLnvoEM4rzQizgv+DzOk59BAxnk19bnA5gMEtNAhoet7RUFqOl4Bk1334Xwa3vg0ZLjihoVi15L/f2IfOYzVllFPgUt2cZhRDeZ4qiSybn63HOFHhF5EE2D+txzDtFeLTXwUxUdtu6TPXvQBGAIQHblyhkRYbeo5r7xggIqEuHCZFAgmNEI0S4YF/kpzv7f/c7xRUFiMUuogTGeiuz1gtXWgvp83OlaVRH/yEeQ2bTJcjBX9+6FtnBhUQdz+wS30nTJ/Am02SeZGOMwnb6ttNXxTqiLCSNgzC9S3HIL8H//VzY92c0Eeryr7OOupbW1K9NbWqAcOVIgUEgyyaOKRrstkkg4+lUXpLbaJpXIZkHMRRUYZl/G4srJTC4zbW0Y3LABoTvvhPz669AXLOD78ngKtq100hS++WZIw8PWQoNZc24azIVvvhn9f/xjbmHCdh+a543ounNhwriPLJM+o/a+WMs6e2tE+zG4WXRIZhW8/tEbQG7/ETK6zLMazPEb1+zAuz+Fx+4LYdczG/DCwANI0rzUeBsK0bByNcXZZ6dxXssBrL/r05AIgzQ8CpLJWG75zOMBrakBYTrCvYeg9vihrDsdob/tROTWG0FGR0H9flBPgN9znZ2VR43yotdqR0dFQrBiI7rJFEcVTs7l48dd7da+nVUDPzDgWHySn366fA38JETmpzQ6LEnQVq8GZs+Gdvx46eyEaUw19o0XFKHC51yYDAoEMxsh2mcik5TOmC9erBTn3bshjYw4BXt3tyMSSDSNt3cbGAAikVw7KyP65W1vR/Id74DS1QW5p4eLo2CQm0T19BRMcCtOlywygdYbG6EcPcqj+rIMvbGRZweMZ0JNKfzbtyN4zz2QTpzg6dseD7QlS5B6xzvGrvfevx/K7t3Inlm8V7SrCfR555X+gvd6IXd3I/z1r2P41luLRqcrraXNX0QAeMqvnM2C2lLgpb4+fpzhcNGUbr2x0ZECbo7FnFSa7dDIWO3QKiB4110I3XEHd0o39hv80Y8Qu/ZaxK++2rEYEbrtNpDRUVd1z0X7wBuRcMZYrg/87t0FCxMkm81FccstTBRrgVjm5+UWHcI3fB7HvvJtRM86D4kE4R5rl/4zAkk/Aj/7GaR4HHHdh2fI2WhX34In/G9H9z2Rsud2rnwCG+WdaKt5ESv/rQ3qxrP4L9hc4Fe1kA8cgFn9TcDg12MIJBMIJBPwqRT0/w6DbeetAqVoFGRwEETXoQwPO+4bZLOlo0YuPgMrFYKVGtEBkyeOKp2c6y0trvZrbWcuPvX25jKQbN4PUm+vtfikdnZa55lEowh9//sTX0s7ldFhSqHs2QN0dEDBDIy0G0zndqZvFCp9zoXJoEAwsxGifYYxaQYkZVZ8WW0tEI1CikZB6+p4v2x7JNDYVm9qAkkkoM+fj8Gf/hT+X/4SjV//ukPk0oYGoKYG0sBA2QnueNIlC5y+NQ00HLaivlIiUVl02TZRDX/zm1A7O7kAlGUeVfZ6oe7dC3X/fiCVKpm6ybxeYHS09Bepywn08C23FH7BMwYyOMjrt7NZSHv2oP4jHyl9jBXU0hZbRABjQCLBj8XwMNAWLYJy8CDkwUHukp6X0q0cPQoaDhecn3yDN2loiDt8Nzae1OQyeNddCH/rW4DZr11RLJEW/ta3IL/2GpSuLssNXhoZ4QIhHne6wReZNDn6wDPmFHhmPb6ZhvzmNyP+/vfD99BDUF5/HdLo6JgLE5ZJXzzOs1ds58B8rcP7ochzm6EK4mot4g0LkOkZQXbbbzH6X+dZDmuMAS+/+Qo8k/0nPPfHKF440QqNKUAW/E8eXiWLNwf2YBN9DJvknZjvOQ66YD4Sl18OVuMD2bULrLYW2SVLICXi8CMBP5IIGH/bp6KMKdCbm0EyGX7dh4dzix5mSQsAkkiAyDLUPXsKokauPwPHIQSZrpc3oisi6CdDHFU6OWd5mRclx29sV3TxCXBkdCh796LxPe+xWuWBUv6MeL2gzc0TXks7FdFh6945eBDQddTJMrTFi0vvvwrq/U+K6drO9A1Cpc+5MBkUCGY2QrTPICbTgMSx4gs4U5wVhTvtptMgRj26ww3b1mcdwSDkQ4fQ9Na3Qjl8uEDkyidOgAWDiF17LfT58wsnQi7SsculSxadQK9Y4YgWFUy8bBMzqasL/gcegHLwoGOiakaamdG6jqRSkPr6oM+dy6OFZop4IFBwbkk6DXg8Jb9IXU2g9+/ndau2L3gSi0E+ftxqQ2Y/Pyd1T4yxiEB0HXpdHUa+/GUusE8/HS1r13KR7PUWjt+o/86uWFH4XpVMKjVt7NZ/mobQHXcAhrlhvhEdNA3Bn/8ctL6epzmrKjAywjM+uruht7Y6hHvJyIWmOdKozbuTGecp9P3vA9/9rrVYpS9YgNjFF4POmVN2YcI06dMVBfLQkLNtoSxDr6113Evq7t2QX3kVsfAsxPR6xKkfGabmxhUmkLu6kHj+EJ6KrsCuXV7s2uXBwIDp8BYpeqoX+7rRRtqxSd6Jdf49UOa3IHv66UBgAxItLaB1dQj8/OeQu7rg0+Lwq1n4WmpQc/xvkCVbrX9e5jHRdf75EQjwxUCjPtzM1rGuFSHcwHJkhLcwM6j0M7ASISj19pZMjbfGT6nVw97BBIujSifntKGBPwvl6toVhW8HOBefikT4mHE9lAMHoM+ZA+rxQDl0yHqWkUjw6yXL0FtaTr6rgcFkRofz7x2EQmCxWMl7Rzh0CyabSp9zYTIoEMxshGifKUyyAYm54suyWSjHjvHJmRnZ83q5K3wmk3OGN821jHp2EALFrA/U9VzNr8fDhW8yCSmd5u27YjH4f/1rxD7zGccY8idJpmCWu7qc6dhjpUsWmUCXmlA73jOR4OnUkgS9uRl6c3NuosqYI7PAFNNyXx+ftCYSfKFh3jxHlBnZLI9WL13Kz6fxxy4+yeBgbgJtXGu7eDGjt/Lx4w4RLHd383Off/jDw9CM8zyee8LNIoLc1WUt8ATuuSdnOmikgJuYdcCQZaidneMWNsXS3cNf+hKv99+40ZrgK/v3QxoZyZ1HE9MY0fg5M3wP+AmTrO4Ccm8vtGAwF/HNnzStXZvrRFAE031cPnHCuVjV3Y3gL3+J0Y9/nJsSliC7ahVoQwPUPXty5xtc+xJKIQ8OIrtyJVJnrEJilEA7mMBguhXZUBOg5867xiTsTi3Fjtha7BxejZc/sxKMlY4h15BRnKvswkbvs2hTn8FsHAcNh8E8HkjDGcgvvQTPCy+ABoNQZYpgcgB+rw5fcwiSj9+T8qHDIHqW38eGCR/R9ZyQNJ+LRIIvbtnPYX42h+3/pvHbeD8D3QpB76OPljw/+dul3/52V9uWZIwIbsWp/Y2NoLW1PAOj2L0py6C1taCNjWOPzfQ3APg94PfzRUmj6wcxviNMs8+J6GrgYDKiw8XuHSNrpdi9Ixy6BVNBxSJcmAwKBDMaIdpnCJNtQELr63mt35EjzkkfpTyiYooYs+40m+WCzIqosYLJNijNiUrDqEvp6gJTFHg6OhC57jreU33JEqQ3by50p0+nIWkaj2ob0S3Tzb7A3XgcaYz5EzPJdLWmlNdpA1Y0CUaEmdld6RWFC2hw4QAzk8DjsaLvlnB8+WXUf/jDoOEwpJ4eSKmUw6TLFPNWtNEu2o3j0GfPtszNzJ7QRdF1vpgwa9a47glHFM44HwWLCNksIv/6r3wyH49DGh21FhdIXsq43tjoaJVVKVa6u+mhoCj8GkWjCN98M1g4zMWxqlpicSyszAmfj096jPZcJJ3mpnGBQNFJU3bFirLp0ybMMLwzMzJofT3k7m7UfulLPMrs1qXbWFAAY8joMmIIIZppwbHXeYRU9TYirHhAMhmckOdgZ2Id2o12bKM0VHK3BBRnSnvQxtrRRp7EarIbsiKBJNJcNJ12Gl+EicfhGepDUB/h6e7pLBRiRMuZB7ruAVI6oOugPh/kWCz3nADl+57n93PLd9c3/qa1tQBO8jPQhRAkqVTZ31e6XSlcRXArnJxnV61CduVKqM89B8n+uWPsi/p8yK5cmbuP167lC0KGEHdg+xxnwSA/Zk0DdD3XEs9w2QchY3c1qAIqundWrRIO3YKpYRwiXJgMCgQzFyHaZwiTbUCSXbECSKVKmzAZkzVJ5xP0ohPxYi68+T8zhbwkcbHl8UDduxeeXbu4adn8+Y5JEq2vh9LdDZJIgAYClpu9nXGlMeZFXkgqxWuITWM9M0Ju1GgTTSvobW3uRxoaAlMUjF53HYK//CWvfbefR7OmenDQSvVlssydzCm1RECBCDczGSjlixXr10NbvhyRT3+6sNY8/xwnk7ma+PFOpEstIhhjk0+cgD5nDq+xHh11pnLb9iH39/OFmPHU2Znp7rpuCeH8xSESi0GbNw8kk4F89Kir3VqvJoRnVHR3585/NsuFSJFJk/8Pf3DlNk2M+4apKh/XiRP858Z4QUhRl251925IAwPQ5syBNBRFKi0hxoKIkxAyviD0UAhkKAX1wAHET1uOv42swovkRjx1ZDleoUvKjqm+Xsc5y/vwlpfuRFv6L6hjg2CKnIuAp7MA1SExHTXZIfgDQHhoL7z6MB84pYCZ2GE8I0pXV67Fo3VymWOxzv5zEGKVkBSrDy88kcRaTJvsz0B94cIJ3a4YlURwK5qcSxLSmzfDu3MnF9SK4ugKIKXTSG/enBP5q1dDW7aMLwDaPvdgmCkCcGSjMFkuXJQ13rdcVwMAVVEXXsm9Ixy6BVPJeES4MBkUCGYmQrTPECbbgETdvRuSPdU6r17dRGtt5WnZdoFWYtuyvzfSzZnfD8oYlIGBgi8cEotxEWa4bdPGRu7gvW+fNcEFMK40xvyJWX49rSXU7Y7d9kmrrlulAtLQEJjHA/9DDwEAaCjEI7bGAgII4YscttpzwhiPoCoKdxQ3IvYl8Xr5l7KiIP7hDyN8663Fz7N9fIbxXqX3RHbtWiua7sBcRDD+q8+axe/FsSLbug6kUsVr2seY0Pvvu49H8SWJC8Fi9xilkEZGeK14UxPk118f8xjtzxALhfh9ffw4SCbDjen8/qKTJut+HPMNnBknDsxWcXaX7gce4D/vH8Jo2ovhxoVI+H1go3FI2SyoqoKGatCVnYOdfcvx+G3L8dxrzUilCIB3FR2CgizWevfgzW/z4U1/34TFizV4n3kGkWf/H8ByhoEEDH4pjSAbQYAOw8fS0JAFMgqU1Ejx47XdZ0ySrIUpa+GjxGeB+cyb58F+ThyLQ+bvbC3WJvszMHbNNai5/fYx68Jj11wzrv2PJ73f9eScUngff5x35DA6eYAxLtwDATBFgffxxxG/6ir+WknCyE03IXL99dzEMt/7AYDe2FgoWgHnZ2Kxv21US124497x+SwfEAJ+/u33jnDoFkw14xLhwmRQIJhxCNE+Q5hsAxL1hRes9GMr7b3IxFuKxfhExxS1xvvbW19VCtF1K3JHEgm+H02D3NPD0zeNPuuEUtBgMDfB3baNv34caYyOiRljYKZoN8z3rH0Zx2lP/Wa2aBQAQFGgNzXlWuPV11tO9QC46MmPKtoFiguYJFk14XTOHEcdtmM/tmtARkeRXbeu4nsiu2oVYLTcsvZpH7fxM1M4kfx03CJIySSPTK1ZYwl1zxNPwPfQQ5BPnLCuc/6EXj56tHRmhw0rOqiqY2xpkJcSzIJBS7yPfuYzJY3i9Nmz3e3fJP+8mOUOtgilvu81xHbsxfCi1dBTcxGW5wJDWUijg4inFeyib8aT5Hy097ThCJvH97O3+Nu1SsfQpj6Fjb5n8aZFfSAfvgyZ9esB8GeVt9bT4ZUyCEijCCABH1L8EssURDMWj8xn237flnu+Xd7HIMTKtjF9L+xlIPb6fYDXVJt12JNuwuTxIP7+9yO4fXvJTeLvfz/PkBkH447gupicW/tubuai1CifYYpi/T9/35m2NkS3bkVo2zao+/YBmQzg8SC7fDmk4WGeSWN+tuh6rh2e4V9hXUvjd46uBphc49RKse6dl17in+OGZ4ti82zJnnUWsqtWQd29Wzh0C6YeIcIFgjc8QrTPFCqtfXKbkmhsp7zySi591estaSJlpYgrCheN2Sz/tyzzGthUyp1wt/VDtlIzdR3y0aP8fU2ha0wMLRdzYyw0EuH9wwFQv5+nHMuyQ0gyVYXS2Qm1o4OLRfthG5EXaWgI0vCwNYkjRvq/WbuuNzRA6e3lE2Cvlx+jLWLO/H7os2eDhUJ8fMb+zMmtVd+df07yjNHGOldSMmml1tO6Oj4OI23eYQ5oJxgclymN2tnJ761kMjdJLzImyy0/Hh97p5RCffZZEMMcT92zh0/wGQPz+aA3NwOqWjChdyuSrXsDcLV4JA0MgBoTdvszNHLjjWWFhLakfAq6NR7bc+iAEDAiIcECiCOIuOSDphGMPv0qUnNWgy1dhk7fWjzVNR/taMMLWIcsPMh3YTfx+RjWrs1gw4Y0NmxI47T4q5BGPGC170R22TLrnvB6Ab+foXaBgrnSa5CYDkbyxKf9/jF8HBy/M8VaiQU9x8/s0XLT0NAot5CPHeOZDGeeCSkahdzVVRgdNoWUrQ7b8Rl4/Dio38/fg1JIySRYTc1JmzCN3Hor1M5OeJ5/vuB3mXXrMFIsw8UlkxnBdezbWFCzX6FS+y4V4fPs3On4rrFEunn9zewkQvjzW1sLwlhOyE6ycWrlJ8goH2hvdz6Txj1JjN9DkoRDt0AgEAhOCUK0zyDc1j65TUl0bGfWQGezPG1blnlE2dzYXo86MsL/b07sKeUGYIAVkXaQN8E36y3NaCfz+ax0YZLNWgLMjCKTbJZv4/Pl9uH1ciGbSnGRbI7P2Ke9Br32c5/DyJe+xI/d1lKOBQK8X6/5OrNFGKVcEPt8gN/Pa+nTaW7KZLasUhTQpiY+STUndUa016znHat9VMnUU/v5MiPqRu08YDhFh8N8scGsuTcXOkxUFSP/+q/jimRJg4OWgZzc1+fcr33MxoJEQcu5Enj/8heEfvpTkNFRnlEBWBEt+dgx6K2t0GfNckzotUWLXO2bmULIRZ008/l4F4PBwYpNfKShIVeLAtZzZGynQUYcIcTkCBIsBApTqFBEWQiP7luCJ78RxrPPejA4+N2y+16qHsL6y1pw9tkZnHlmxhH41bAcAA8Gh30Mfj+F38+sxAK1tQ6orQGGh/kYZVtNu5llwhg/L36/syzELBuhtDCFnFJHtgOhFHpdHYgsg3o8vL0bY5CPHUP8qquQvuAChzhELAa9ocGqw5aSyZImTPErrkDojju4g7kxLlpTg/gVV5x05DZ4113wdHTkModMGIOnowPBu+5C/Oqrx7XvyUzvP6l9F4nw5X/XmF0hCCHQWlsBMwNJUXh3hJ4eh5CturpwSuH/3e/KbuL/3e+s8gHh0C0QCASCqUaI9hnGWLVPblMSC7arqwOJxXiKeiaTm8zbYD4fd1kfGuJRVkPoAlyomuKSyXLJlliAYdIVCOQml/aItEn+hChvLNLQEHdJNl2MFYUL6iLOzur+/ai7/nrErrwS3scftyahZqTXNHsrGKemgcTjyJ51FmKf/CRYbS28jz2G4A9/CH3OHEeKNYnFLMf5k6KYcR9gtWxSOzog9fdDmz+f95LXdcvYD5LEr4mqIrN2LZJXXDGuIZidBOTBwcJIsT07QJa574CbSDsA9cABEF3nE3ZDLNvTxOXeXminneaY0KumgBoreyOZBIJBpyma3UvBXj5AKYa/9S1AUSo28SlrAGiDKQoymoyYHkQcQaTgA1M9gCRBYzI62Jl4km7EDv1c7MYqsIdKv3ctojgPO9GGJ9GGHWjGIKIX3sn7phsoCo+km3/y29ebWC7jZpqw6RNgi27r8+dbmSwWZqaLJDkX85BbnGJeL9+f0QlBNlKlJULAhoZAa2vB/H4u2O0p2nZxqGlAmUUUT3s7T19XFGhz5zpEfnD7dmRXrRq/cC9iemim6wP8My50xx2If+xjKHmCyzCZEdzJ2Hf+d43c1YXQD34AyfzOCAb5gltPT4GQrba6cKuNZalnlzEo+/dbWVnCoVsgEAgEU40Q7TORUrVPblMSzzmnYDupr88hnAtEtCzzVGXTbfvIES7UPR6eKplOW+2ACFBeaJlRbMasllgANz6SUimrL7o1frOe3GzFRSnk3l6eDWBGl4BC4zTbOKSeHtTceitYOAxaVwdGKRdgZc4xDQQw+vnPI/ne9zoEXeBnP+PHblt0kHt7eb27vdZ/LMw2b/njLlZDDqDm9tsh9fTkethnMtxxv6GBi4tMhkcog0HEPvnJcUeCxuwkYCD19vKFEzcu4OALG3T2bKfpn7Fowox0e5JI8NpY+4TejHqWyVwgjAHJZO6amq30GMuZ/pnlHEaa73gifMTsAV8EBiCBAGIIYVRpguYLQW9ogDQygr5eCU9q56MdbdjJzsUowqXfAxSrSQfapKfQBt6OTSLUSgUnmgbv3t3wrF9uiXTXZda2KGKp6PbITTchc955UDs6UHvDDVAOH4blCG+aMxqLI8znA21p4QZzXi8X3sbCmSOKn0xCTiSc6e4GlZitWZ9bxmeR+XTotbUnnXJdyvTQ/DxjkgRpdBT+++5D8rLLKt7/uCO4bkqdJis6nPecaIsWuRKyk20aWCnqCy/kOhoUw7i/1RdesEqphEO3QCAQCKYSIdrfQLhNSfTfdx+UV14B9fl4dH10FLJRL10MJklgoRCPLBsRVhoIgCQSloCELPOIt1kXDjijnvbopySBjIxANmpd9dZWoLsbdNYsUEK4eMtmgXgc8tBQLgputuIyUrb1ujpI8TiPsBcTjmYtrbmokMlAN6J90uho+VpySiHF41b9uEl2xQroLS1QDh0CbWrizsPJpDUJJUaUzqozLzImSwxoGl8cCIWsrIGCemHzOuo65Nde4yn59h72xqTc6s0sSaCKgtD3v4+YJI0rIlTQSaAYxrUuWfNeDMZyaeyGh4El4A3k7m7QxkZrQk8jkeJO9nmQZJIvKLW2WtkipoEgM+5D83znG2adDBpkJBDEKEJIIAAzBj301ndj15L3Y8fRxXj2cYrXUFOyLh0AGhp0bNiQQRvaceGDX0REiTnuOwmAnyQQwCiCGEGqIYrELBcu9kVwG93OrlmDkS9+0crIcQj84WFI8TiYx2MJduu5BYoLm3JmdZWYrY0n5dqF8C1remhmAxHiuq1gMSqN4Fbivj4V0WG3Qrbq6sLdLqTmbyfMwQQCgUAwRQjR/gbCbUqiZ+fOnEhnbMwJDSEE2rx5zhrg1au5idShQzxFO5t11lUbQowZ7uyEMR4NZwzarFmQUimMXn89suvXA5Si7sorrXY8SCQgDQzk6tKNfUpGGqY+dy5w+DB3wi4X3Zbl3EKDeSxGJBpjiEAAgK7nIreUwr99O4L33AP58GFI8ThvDebzgdbUWHXwkGXeduzEicIItNEWC5IE6vUiedllyK5fDxqJIHLddcUXEuxu7ZFIrm+y3w99/nzIr79ume7p4TBYfT1INntSDs1WJwHD7C4f6+eyDK21lQs4o5f2WJjXmMly0VIGs749u2oVn9C7aK/GFAXRrVtBm5ute4lJEl/wMQwGYZjO6bW1vG/6a69BHUfkjIXDSMGLOIKIIYQ0vNwMEcBrOA3t2IR2bMIzD25EWiv98asig7U1B7DhAhnr/74ZixZpIARQ93kQ+XMMkq7BK+kIsDgCLA4fSYMQAkKzYB4Fo2vXuh5zMdyKL0sIbtvGjR+zWUBVkV29GukLLrDKTaThYTDDX4HW1UEyFrKsc+/zQY9EIA0M8JIHSaq8NGGcKdduha8r00PGKu8gkEemrQ2D55wD/333QT56FPrcuUheemlByv143NfHFR2utI+6GyFbZXXh0ujohG5XQBX0ohcIBALB9EaI9jcQNBLhUbDBwVzNuC3CQdJpQNfhe/TRnEM6MGY7Leg65BMnMPrZz/J6V6Pmuf7DH+YRWTPaWsohXZJ4kNFYICCGE7F+2ml88kdprh1POu0Uc7bxS+k0qKJwEyQjHd+MpIOQAiFuOXjbx8UYzxBwWYdNBgfhaW9H+Oaboe7ZUyAiSTIJOZWyaoL12bPBAgEu2vOOwUwXJuk0EAxi5GtfAxSFi5i8FmSFAyEF7cxILAYpFrMWNpSBAbBYDHpzMzd0O34cNbfcgliZFmYlMRZaig7FPAe6DhYOg2WzgAvRrre0QI5Gobe0lBfjjPFuAOC1qGOVGxBd5xPlvHspt0EuW0Hp6QFTFNRs3eq6ZzSlQDJJkEgQZNOtqMF8AEAMQTyNc/EkNqEdbTiKebkXFRny3Lkazl3Wg3NO68batVmoq5cYY+Mbe72Af8NStCwLItz5HKR08ewRbenSiYn+VRhFJMZ5NP/OrlqF+FVX5WqeX3uNn9dgkJdXGOO1/lZVkGgUtTfcYAnwSvp2jyfluhLh69b00O12pSi2iOD/zW+c52CK3Ncns496VdWFuz1H4ziX1dKLXiAQCATTGyHa3yB42tsR2rYN0ugor90zXH315mae2s4Yd742UjyZ38/FYzlsRl4kmYTvwQcxeM89gCTB+5e/QBoZKS3YTTQt19fYNLwyWsZZk2uzHc/OnYVR/zzBLcXjkPbs4duZbYgAq8dz0fHbhX80CtLf7zpd0vvYYwjdfTfkI0eKC01jkse8Xt7rOxgsdFM3W2XBSIk3MgDMvuvZFSv4eaTUSh+36rDtrty2CCOJxaB0d+fGJEn8HCSTPMW8oQEkkYCnowOR664D8/tdTySzZ53lzmzN48m56o9lFqeqiH/0o6j50Y8gG34IJTFSkNWODvgeeGDssTAG3wMPILtuXcG9ZJqnEUqtxSBaVwfa0lI2apnNAokEF+rJJLG6HXbvGMXfcDXa0YYXsRYaSveF9/sp1q7N4uyz09iwIYO5c3Xwj+SFAPhj4fPlzOPMdRvp7y+GtGdXyWMliQQ8O3dOiSDIF71mtFTdt6/gvKnGYp5y9Ci/l81Wjsbnh3LkCL+W3d2OMg+3WSEVp1xXKHzVF190dU7UF1/kGUITcD5LnYPxlgJUIiCnoo96tdSF6/Pm5Up6SiFJfLsKqKZe9AKBQCCY3oj8rDcA5sRB3bcPelMTT7Ok1BJw0sAA5BMnuDGXLIPW1fFoZ7lJTN5EkYZC1iQRQG4BwC7Yi9SsElvNs1nvLSUS0FtaIPX38yizpsH7+OOOlm4lYSyXdk4pX3jIq4vODbrQ6I1kMrkIvAvUV14BiUZz75nfCsrcr6aBqSrkEyd4OzN7X3lzG10H83p5yyRZttJ41c5OS8ibtfnmGK2otlHrb54DubfXkX5vHj1TFO7Efvw4354xsJoasGDQmkh62ttdH385SCYDqbeXp7urpcUrAGitrUh++MOI3nILaGNj+R0bbf7UF16wWsONORZzO0rhffxxnkJtayNonStJgmRmRvj90GfNsnrHJ+MUg4MSurtldHXJ6O+XcOyYhIcf9uHmm8N473sbccXvP4Lb8Rn8DRuKCvbTsRcfww9x1zt+iT/8oQ+33BLFe9+bxNy5OhQFqKlhaG6mWLBAx7x5OpqaKEKhnGAHpQj+z/+UPVbl8GFEbrih8DpSCrWjA97HHuPP1VilBWNtnyd6mdEXPf+8ma+zFp+MZ8He25sZ3R3AGPTW1jH3VRQj5ZqFQvz+HhoCGR4GGRqCfPx4Qcp1JcIXAORjx8qfLwO32xVQwfm0MhHKlAIQTXOUAljfA3v3ggWD0FtaSj/3FV7bk8LI6LC6BpyCtPHkpZeChsPFP7+Nn9FwmJcpuGUqz6FAIBAIZjwi0j7TKRJN0jwe7mZuGENJfX3InH020m1tCP3oR3wiKEnQWlv5dkYqclGMfuzU6NltThItgzYzYm2PaucLaE3jqdaG2zMZHYVy6BAin/oUmKqCtrZCPnIEtLYWcirFW8eVqzm379+sJS+CtWBgtEOzJk8VTKJIKpUTpPbJnt353BDbyYsvhnroEJTOTkerNucOidWL3sw0MPuia62tkPv6CmqBYbZ1M861ZXxHiJVdYLmi28+P6QXQ3w991qyCHuilJs/qCy+4OzeaxhcGurpcm9Fl2toQ/9CHUPuVrzjOiYXduJBSHvV3gbmduns31D17eNmGUVJA7L4NxmIWSSah+4OI0wASoSVI7Mti6PFDSC5ejj17VOza5cGuXV4cOKCAsdIFJLUYQht2YCOexEbsQAt6AQAj67+IjM/Zhs2Nw7v6/PPO0opiUAoSjTquY6Upum62rzTa61h8MkwArUi77XOiIL2dENDaWiidnQj89KfIrl9fMhpbSZ/2qms7VsH5rLgUoNKsgmrroz7ZKApi116L8Le+xct6FAVEksBM41RZRuzaaytq5feGO4fVDqVQ9uwBOjqgAMiuXCl8BQQCwbRCiPYZTrGJAwuFoBlp2mR4GCSTQfKyy6AtXQr2059aE0FzO/n4cUh29/g88ac3N/MItW2SSBsbQX0+XlNt27ZoFNo0aDOd1il11JTLQ0PcDX7WLOcE3wXMFAmUOtuISRKo3w94vdDmzoVy5Aio3w+5v7+oAVrRfasqn8SVqze3iVV9yRKMfv3rUDs6ELnmGihdXXwT+wSxSOsra4JuF962v2koBHloCFI0CqqqPHdb1wsXK0q41ZN0GnJ3N/TWVlfiSD5+3N35CYV4PbthRFgUI4olHz3K048VpXD/JQQ/C4ehLVniaizasmUA+AKFNDLC07NNvwNbRkIWCmJ6CMOpRiTkRjAQHKP1eGpkGR7/xizs6mlAPFn6Y1OSGFbTF7AJT6INT2IVdkMB5YtRYPAjiQASUC9vg6/WXSs8O4G773a3oaZZgoCMjPAU3dFRUL+fHzelUDs7i6bouk3pdYheI8UdhsBhfn+B6HUsPvX3O1o3MlXNLTBpmsNIn8RikHt6QJJJ1Hz722DBYNmUbrd92h3C1+crGH++8NWbmlyderfb5VPJIkL6/PMrKgWoVEBW24LGVBC/+moAQOiOO7jhnKaBGAtGsWuvtX7vljfiOaxWrEXIgwcBXUedLENbvFj4CggEgmmFEO0znFITBxKP8yi6YT4X/vrXkV2xArShAfKJE7mJICHQ58wBk2XIPT38xaaIkmXozc08yn7iBJ8krlgBtaMDnieeKN6f24xsG+nhLBRC/J//Gdkzz0T461+HcugQ/11+FI4x7tTutn2YeZyGyzkzTbI0Dcznw8gXvwh94ULQ+npIvb2IXH89F99mj2l7lkAJWDgMxpijNrfYhJgYAjG7dq2VCspqanLnw1xQMP/k7SO7ahVofT03ujPPjWkSaIh8bcEC6K2tUA4e5KngpmC3pR0XPQYjTZlkszytV5bHFEdu3bGZxwNWVwept5cfkyznzA3NmnxJskR95F//lS/auDEBVBTQhgZ3qfySxMs1YJRtUMrLCwivRU/CjzhqEUcQGahIwYunYxvRHj8fO0bPwmvaAr6fw8V339TE27Ft2JDGxpGH0fpfX4JZkOBDCkEk4EcCfiStnt7Sn/9Y2MvbTduxI0fGPl4YWSSaBqm/H8H/+R+QwUEQXYcyPJwTyl4vkM06MysqiMiaolcaGoI0POwU4V4vaG2tcyHP2B6qCm3hQpDhYUiZDKjHA3g8UF5/nb+XLZpJYjHI3d38eCSJl05IUvGa4Ar7tFs18C+9xDtc5I2fyTKyZ51lCV/ltddcnXu32+VTUfS8Qvf1SgVktfVRnyriV1+N+Mc+Bv/vf4+64WFEa2uRfPe7K4qwm7xRz2G1kb8IiVAILBYTvgICgWDaIUT7DKfYxME+EWaSBCJJoMEg1H37LEGYPxEklIK2tACpFIimgYZCoI2NvD3WiRNgoRDSmzej/kMf4unHhjgqiZHanV25ErHrr4fa0QG5q8uKutkjyUxV+aQ6k6k8nS0/lV6SwDweZNetQ3b1anja21Fz++2QRketCCwAR2p5qfFLQ0M89d1wPbZ6r9tLAIz/a8uWWSmQ6u7dkAYGoNfX87ZjxvjMdHm9ro63vjKiXp4dO6AcPmyJemKmxpsRfkLAQiEMbt8OtbMT6rPPovbrX8+ZfQFFsxOYLUuASRLPMDAWLMqJo+z69fx1xRZl7PsPBLgwtNfhm8Z89g2NRQv5xAm+QOT382tRZoGGBgKQu7sR+vGPy47BPD5TtNO6OuhEQVz3IybXIc4C0CHhEOahHRvxJDbhb3gz0tHS/gkqMniT92W86eIA1v99E047Tbdu17qfv4Y6DCJgiHTJ6MyefyT5vbxNo0izbRpTVWhnnIHYpz7lXDCZNw/YVcKEzgaTZTBFgTQ0xJ9HYyEn3/yNSBLUPXtyKeyVRGRXrQJtaMgtJuXtOz9jpJRIlgjhfhpmJwlTWBreDJaPg88HFgjwhcSxUrph+BjYIucF6ch2U0IjJdpsAUnicRBZRnrz5pxJpMsMHLfb5VOpkV4l7uuVCkjHWLxefkzmufT5pr6P+lSiKEhddhkwezZSx49XvFBsUnW96N+IFFuENH0FJrDLgkAgEEwFQrTPcAomDoA1EWaqakWeWW2tFY3SW1rAFIWnbzPGxbUxEQRgTRLlvj5rkpjevBnB7dtBRkZAbOK3KIzxNN36eisapL7wAh+LOfG3YwgCsy87UxSHgV1ZzNeCCydCKaR4HJ4nn3SkDTOPxzHZJmMJUkkCYQw0HIYcjfK6aLMu3j4uWYbe1ISRm25yRr3icZ6Oa0TqHeMbGQELBCAZ7eQin/kMSCLhPG6jVp/5fNAbGrjIN9zmpcFB0GCQdwooJyBsLe+s4zWyAkgsxnu7NzdD7u0tiFAyv7+814GxKGP1XU+nrUWi/IwCc0HBMmtyc10zGfjuv9/pMp9fs2++lywjVdOAoSGCLGtFb+gsxEaApzLnoJ21oZ214TjKZw8sxGto8/8N59Z34k3+l1Ez0A3t2GLEl34H/iCx6tJDp/sRQX/ZfeX38va0t6Pu+usLMknkp5+G+sorGNq61RJgiX/6JwR++9uxz4+iQFuyBLS2trAcwDhX5kKTFI1CffZZZFetqjjlvSz5z3E5kZxI8MWiUAhyTw8X3jZHfzOrx1EWUiKlm2WzUI4fL4ic642NTnM205QwGOT+C2Z7SkL4gpOiwPv444hfdRV3Dl+4cOxjBlxvV8A4epe7dV+vWEAaY6m7/nqo+/c7n0lCQBsaJq6P+kztYT4Rvehn6rmZIoSvgEAgmEkI0T5TML/c+/shDQ3xtlVG7+3YJz6ByGc/C7mrC9Tn4+ZphnM2JAl6U5M1OWcAlP37wSKRXO33rFmIffKTlnAomCSuWIH6D32Ip+BmMuVN4gCAMWgLF2L0C19wn5ZmpjObrduMtOqxMIWB5Z6u6yCaBv+DD8L75JNWKq3U3++6VhuA1ZqN1ddDC4d5ezVzXJRatebZ1asLoqU0EuFRTtM4zhDixChHINksEItBfvVV+H/3O+4LQAhv02fUIxPD+R6KAhaJQLKbANbXj93X3XhfZmYF2BZZSDqdK4U4fhy0rq7AUIx5vTlBV3Tn3M1eGh5G9owzuDDcs6fkIgLzenkUFUaEdAzhLqVSUDo7odfWQonFciUAxjlkjCFJvYhJtRgNzkZvphV7dnrwzDPr8bx2NzroYuhlPv4CiOFcPI1NeAJtaEcruoEkgEwDfOEAgnUZBI+2IzbyErILc5O9int5U4rwzTdbJQSOaLWmQertRfjmm9H/wAN8wWTtWsswsPTJkcAiEcSuuYZfR1s5gIXxHJjXvea//xu+P/8ZqXe8w3XKu5kxos2Zwxeu7Nv6/dBrax0ZI25Est7ayifSBw/yMglK+b5aWnhrShvFUrqtlnK6niuxAb+nlKNHQcNhK5psTeibm/kCk5FFxBTF+r/9vo9dcw1qbrvNlYP9eBlX73Kj5KYs4xSQjqfQttA2vthzITO9h/nJ9KKf6edmKhC+AgKBYCYhRPt0oYzzqfnlru7Zk0vxliTQcBj6nDkAuBCTEglusEMpNz4LBEBrapyO5GYqajoN2tTETcK6uxH5/OdzKdJ5k0S1oyOXguvSJE7dtw/q7t25lOu1a3OR/yKC02wHlz3jDCivvw5WWwuppyc37mIYddSOHxm9zuXubkiMcZd7gJ+Xsfr0FoF5vYDfD23BAkjDwxi99lqwSMSxaOKYCFMK5ZVXrL7rjrHa2uMRTUP4O9+xIuz8zVguvc8YK0mnQUZGnKmtZmutcosaRhqyo91ZMXQdUn8/j9znGYrpjY2Q+/sLXs9kGQSA1NfHsyk+9Snu2t7ZWfJt7JEQYqSyjwWJx0Fnz+ZR/2QSug7EWQBxBBFHEL1oQrt8IZ6Q3okdnzsXw8PmdVhedH9nSPvQRp/AJjyONeRleFgKkmUeF0cACXgHDkILzAMUBZKWLJjsVdrLW+3ogLJ/f+mykGwWyv79UDs6kF2zBpAkRLduRf1HPsIXTYqQXbkSIzfdhExbG7x/+YvVh94SWqYxoG1hhAUCUPfuhdLVBeb18vryMVLevU88wSfELS3Q6uoKRC8Yg9Tby88RpfDfey93Pg+HwWpr+b2bJ5KlgQEM33ILz7557jnUbN3KBaablG7zvs9mrXZ+9oUc06Axu2IFgLwJvbHQYBejBRN6ReGZC2XuT1pbO676ZzuT1bu8IgFpphXrOrLLlzuvldcLuafnpNOKq7aH+QS7jI/nelbtuZlmCF8BgUAwkxCifRpQzvkUAP9yHxzkotmsU9d1Hi0bHOQ1oLNnIztrFo8o9/ZaqajSwAAXxGZNtoE0MgIpHudRrqYmSEZP2WKTNKm/H9LwcFnDs3xIJoPwN78JgJv/ZFevhrZsGdTOTsuJPr8dlLZsGUZvuAGR//gPPpkJBiHnC5f8Wmvz37ZacL2lBdLQEAjAJ0NmizRZ5mnuFdQwklSKRwl9PmBkBPrixUhfcEHRbe2LK2Ol3wPc/IskElZEtKAvs3F+pJERh2GW2tkJ2NPGi+6c8Ro/w6xMGqPfuZRI8D7GyEU05WiU70pV+bk1jsk0DdMWLcLojTcic955CN15J2goBJLNOiOyisL7uY+OghrpzySdHvPcmMeQTTMMNSxFunsIwwjgRazBkzgfO9CGTqwEsgBKBFHqMISNaEcb2rGR7ESjZwRSKgmvoiEgpRDMROFDiutoq00eoBw5Yi2cyEYHABO32RrmdmOWhRjt0dQXXuCi3YDW1kLWNL5IZlvMoZEIRmwZLLSxETQc5lHzbDaXYZNXwsECAdD6et4polxJhW2M+RPifNFLUikwRYHc1YX6D37Q8rqQRkfBBgehRyK5WnbYRHI0ynt2r1oF3yOP8JRuj4d3ujAijswQz9q8eZD6+3N95Iu1bbQ/04Zrfnb16oon9GpHx5gmiSQezy2wnAxuoufjwK2ALEgrZoxfW7OF3smmFVfYgm6qmDSX8UquZ5Wem+mI8BUQCAQzCSHaq5yyzqc33ABaUwMyOsqFkl005dUgSsPDoPX1oM3NkGIx7h7f1wfAcBAvJiLNFmRHj4I2NUF55RX4770XtKHBMdnzPPOM6wi7A11H6I47EP/YxwBFwchNNyFy/fWQjYUEuxjRGxqs6GH0llsQ2raNv6/d9M1MoTeiEgAKap6Zx8MXJzweMMYs93zoesWC3R7NHmvF3n4dWbFIXIl2bDCc7wnAr4d9QcOI1rNg0OkU3dsLqUQU1v5+Ul8fWE0N9NZWSAcOjLm9cvAgN/AzI5qaxkUXuHEfFIWPUdPAgkEM/L//B3g8PJr86qtgoRDI8LAjmgxV5dchlQJJJnmKfJnUfgbu9h5DCKNqC147fhqeUC7EU/qZeApnI4FQydfK0HGWuhttZCc2ZR/FSmkfWG0NfH4CPxKoGTqGAPoBogKyAiKlAcoKc4HNxSTGELrrLjAAdP580Pp6ntrvArcO/AW4iYB+73sY3LjR8h/IrlzJzd80zZmZYj4vfj9fdCIE1O+HMjQEvbERUiJRPuXdPiEuJqqjUeizZiH0gx/wrIhgkEd3DaM3JR7n19rI+sh3m7fqqq++mi9E2Z+R7m5AUaAwhshnPgOmqrwTRDye8zIAHCndIARSMmm1r6x0Qm9fYCH2BQJjrOaCSP4CS9XhQkBW7A9QIdVYa1wtLuPVeG6mLUXKQswOKa59BQQCgaBKEKK9mhnL+bS7G8qxY9Cbm6GUc2s3xLcpivSWFp7+appBlep9brTlMmtrCaUI33STlbqrGQZ0gZ//fNyHKA0Pw3/ffUhedhky552H0euvR3D7dsjG+OD18rr8a6+1JkyZtjaMhkKo/8hHuOCLx53pvqpq1a6bbujMdIBOpyEfOwZt8WLos2ZB3b+fR5Dtgt2+EOBi/HooVH7FPu86WjXq5d7DFFOGc7NpxMc8nlwtsmGEF/2v/3JMJtWXX3Y1/tTFFyPx0Y/C++ijUL/znTG3l48d4/vv7LTa4xWNihsO/eq+fTmDMMN4z8zqMM8LSSRyGQN9faDNzTwDZGTE2p0OCXEEEUMIA6jH0zgXT2IT2hPn4zBbWHbMs8gJbAy/iE3yUzhH34FQRIK/txtBbQgBPQZfNA2JhXiddcQDkgBfjDDc10mxjAXGAEXh7v/HjqH2K18BC4f58YyOjnkeAV4OYv49ZlmI2S4QeRN647PAfqVpbS2Uzk4EfvpTZNev58+OMWlFLMZd9wcGclklkuQ0eDMFr98PraWlfMp7OVF99GjOxT0e559fAFhvr9PXwDSWLOI2bx5vyU4CmgZGCGhLC0g6nftMk2XA6+XXw95aUNO40aOZ3j6eOu/8z0qbJwQxvCxmAg5/gCJdB/L9ASql6mqNq8hlvOrOzTTHURZy8CAwOgoiy658BQQCgaCaEKK9ihlrxZ0FAjwtPb8uOd9FG7BqPU2TL722FvLgIJ9olokwW2ZpxoKANDJiCTZ550543fTJLgdjkLu7nXX5Zqo9IaCmkKEUakeHZbQnHzkCGGntxG46ZYh4R72u6VBtSwtWXn+dp8TKMnfttp+7CqLt0uAgkEgAtbVIXnwxH2M06shEyL+ORSPtxU6N4d6udHfnHNabmoBMxnKYj373u8icf77r8drRFy5EdvVqeB991N0LzHtgcJBnVpTyADBMAi2DMNN4zxCgjmthWyjRGxshxeMgqRTS8CCOIEYRwm6swpM4H+3YhL/hTcjAbAlW+NYepLFB+hvOq3kJG+t2Y5H+CvwnuuD3M3hb6xHMROHVDoERCgIdoACGhriQs7npg1KeNm52LbBhekFIAwOWMNRraiAfPz62CWMebstCzIhaOXd36DrvDJFMoubb3wYLBi3jKmvSavgKEKMrhN7c7DR4M4Qns0XXi6W8283ciopqxqznmLa05J4r++eY+W/zHsr/jNM0hO64g4tGQ8AQSnMlAQDkoSHL14A2NEA2so6Y2T7OPnbT68PwsQDAFwqvugrBe+6BdOIEbz+nqkUn9NmzznKOz6yVN4635HbTEEc2jVHzDyDntWBkKJn+AJVSbbXG1RTdrrZzMxOwykL27EETgCGcvFeBQCAQTDVCtFcxY664Gy3cHNHOMpEe+cSJApFOa2sBVYXU1+derOo6T4V2UZftBvWZZxD4//4/XpdvGOUB4BP/WAzeHTvg3bUL1Ofjad9mRMsUU4SUFEtMkvjvbOKSyTIIpZC7u7lreSTCBYbxniCEZxi4NKWTUikwWUbt176WExgeD7QFCzD6uc+BGPXo5nW01/GWhDG+nSxDa23lgjCd5pEvs65UVRH6/vcRk2VntMDtdTS2c1vPZ25ninAwxo81L6JJNA0kkbAirQBy96WmFe8ZzxiScaD3ozegd/8o/vabE2jH+XgSm9CDWWXHtQgHsQnt2CjvxNqGQ6hR0wioWfgDQDA1CJ/+KvSa2aCKH1Iyw8dQ7BzZHfQzGacxnLlJJMLryXt7eQTaiJKTkZHiUfkSqM8+i+y6dYAkuSoLMSeX5oS+wN3dXGgwa9sbG/mCkS21d/Cee6B2dKD2hhsgHz0Kfe7cgiiylEzyBYlkEno4XL4/dzFRbd4HxjmURkeht7by/xsdE/LTy4npEJ/nNu+/7z5e/64o1jgZAOuKGCUiUjQKWlfHr5dp0JjNOhdAzEXAcJifGxS6c4MQ6C0tSHzwg0hccUXhhF6S+Oee/bMm/z6yjXU6Y2XTGJ4KjkVRw7MCjCHws59ZGR2VHHe11RpXU3S72s7NjEGSoK1eDcyeDe348cpK4QQCgaAKEKK9ihlrxd3uMO0mQmzWm1oilxCeBqqq7r/AzPcZTw17Cbw7dlhu0kUjt4zx6HImw8WDYbTn+H2p4RYZp/Vao5aazpoFGouBMcbreEsIy3KQVCqXJmu4V3uiUdT/8z8jccUVuevo84H09rrbZzQK1tAAFgxyAa9pjrFLsRi8Tz1V0MvbNIwbC3M7yTCUGwvHduYEPpOxzj8xf26PQBqvY34/r3m2XSsz7X0YYezCOXh88K1ov/1CdCSWgKK0AAhhFOfiKWwMvoBzI7uxqOdZ+GkMQX0U/v4MPMhYddLmwpb1t73eudy5CQQg2e9H41mT4nG+sGSIQBilCvLQUEWLWN72diQ+/nEARvrm1q0IbdsGdd8+biLo8SB7+ukF7QKzq1aBNjRA3bMnl7VhlikY4p15PNwbgJCC1N7smjUY+eIXeUq40Q/dkRJeU4P4FVcg9D//w8diLggY15Q2Nlop4/577y0qqk1MEz0yMADW0pIzqlQUvp2Rrq43NvJOFUAu9R6AbKRmO9zYi2QFmQt2zOezzButhQFbHTZTFCv9vpQ7t9zTg9APfwht0aKCtFkpGuV1+ebCYp5vCCSJ/97l83TKcNH72yx/0FpbIff3F5hHArzjRn5GRyUtPE+6h/kE4viu9fmszgwE4PfUVEa3q+zcCAQCgaA6EKK9ihlzxX14GNqyZdxYyTTAMifYRYSJKViIpvEJeDDITencRggnqV6TaBowMuIqtZgZ0R8QMrZD+hhI0SiPJO3Zwx3lR0d5izxCQIaGINvT5scif7HBrNVNpRDYvh3a0qWQu7pAdN21WZk0PAwqSXwhwXSRL3JdpZ4eRy9vNz3aAUA5dIg7YrsUGSQahdrRAe8TT5TORDCFIyHwPvGE5Whuihwz7f0QFuHPuAjt2IQd2Ihh1AEZ8D9FWImXsQlPog3tWIvnEcYovL4AfJEwggOv5+qkzSEZKfhWyzxzglvOGd3GyNe+Bng8kLu7Efj5zyEfO+aoxzcj4uYiSsFC0hgQW80+MM42X6Zjf76QzUtBz0/tHav1FwDANGjM368tq6eoqLYjy7w7wegor5s3n1td5+fKyAyQBwchxeMFRnRWJoCR1m6NIe/zjamq9Ts9EoGSSvHIfTgMYqT6S8lkTuwA43LnpvX1YIEAtFAI8tBQbqGOEF5qEImAMFbVactue3+bIhaqCu200yxvA2QykPr6rGh7sYwOt8L9ZHqYTzTWd+1LL/HPaGORQjEXfGTZ0aFjsqmmcyMQCASC6kCI9mrGhfPpyE03AZSi7hOf4CneeZEx+4Teih6bLuNjtC8qyiSllLmtBSaUcpMwxjAhSwiGgFCOHuWZByMjjgita4qcF3NxhKTTkAYG+PmuIIIvxWI8XTkQKOwIkPfeyt69VqsptzW1/t//Hr4//al4FkcRQnfdBWzbxvtpjyF+pVQKwR/+EP67f4aRhStwdCSMJ/AuPI4L0I5N2Iczyr6+Af3YiB3YhCewEe1oxVEEkOC90pEGAaCRRjDJX9qAEeDnTJYhDQ1Bt0XQxkJ9+WWMfPObAKXwPfigZcLHD66IkDZbDbosqWDFsiFcuHqru3dDGhiwzPoK6u0N40jLiR/FU3tLLhIAaHznO/lniemIDiMtnVJIw8PWAlFRUW3H+DkLBPjnV20tj77b7h3LIDKRKDCiS156KcJf+hIXLKbgz/9ck+VcGQZjkDIZZFes4IZ8+/bxzxWzRt3IWjC7GVRav2wXdkVr+GOxKRV2lVJJ72/HgvGsWdzbgDEor72WW6jy+UpmdLiNAo93sWqie9hDkpDevBnenTstg1aiKNzQNB4HkWWkN2+e0uj2uM6NQCAQCGYsQrRXOW6dT4fuuguRz3wGJB7n6X2M8Um6Gf0z2nFZfwzXeAdm6nAZcThjYYxP1HSdR5VcRmTd7BfgfgJMlkEqMLqjgQBoOAx5ZGRMMUg0Depzz/FWUy6FI41EAEXhfgYukE+ccDVh1CAjhhBe9l+IP6XOx45dZ+EZvAkJBEu+RkEWa/Ai2kg72uSdWKftQghxBJCAH8miCzQSpaCJRPkFH0JAdB1UVfk1cCuqDcFrimRtzhzI0WguTTjvPaz7x2X2R3qc5oHS4CDI8HDJBTdilq7Yov4lU3uLLBKoL74IZf9+Rxq0ZUpn+BUo+/dD7egoLqpNjIweWlvLU/9/8AMor7xSsMhgb1VZcG8pCmLXXovwt75lHYOjBSIAva7OKnMxFzKT73kPvI89xp81wPrbcQ7HU79sCrv29oJnjCQSIMbvq1JUVdr7u9iCMaW5z0VZdnYdOBmztgp6mLvNFKgYSuF9/HHeRULT+HOsaTxTIxAAUxR4H38c8auumtrrW0l/d4FAIBDMaIRonwa4cT7NbNqE6H//N5/QvPIKT5k3DKJINssn3PaJpjlRtotIu6GUkQZbzD37VMJsTtkTjVmLPinHbLaEqgApmYRUgRu53N0NtaMDwR/+0N3+jXIA2tDgvg63xDEk4UMvmvAoLsJfcSHasQmvH1tYdldz0G2lvG/CE5iFHvi9DL6IB+qJY2VfCwBUVXnf93KLIIaITV58MdRDh6Ds2TPmfkEIUu96FwCbwGtpgVZXZ6UJk5GRXNeBYtktY5Bdv37scRSBhsOQ7OUVxTpF2F3aKzSusnqRmwaOeceW34vcEtWGq7zD+d4Q3ZkLLsDg+efDf++9CN90E19wKPb8EgL52DGH6ItffTUAIHTHHQ4vARoM8lZvqRTk3l4rdTi9eTOC27db0WSzFljdt8+KJo/bnZtS+H/3u7KLmv7f/W7qhZ0LxuOOnp+iTeJx/h3h90NvaXF2HcDkm7VVkingwEVk3jo/zc180TuVggpAA6z/i97oAoFAIDiVCNE+gzDFvf/eexH+6ldBg0Gw2lqe0mi4fQNwCguzNZLp+p3N8vRzQ+Rbvc4rFJyTBWHMUVc74djqlN0NqIhomsjhSJLr6G3wpz9F4Le/BYnF3O3cWBCo6HgNKAhiCOI5rMfD2IInsBnPYT2y8JR8jRcpbMAubMITuAB/xSrsRtCIpCvgYkybc1quj/YYSENDvB/3WDAGfdEijH7961A7OlD/j/8Iqcw5ojU1kIaGoHZ0cOFnE3hmCzQiy3yhwzh3Zo0v83onLkujCMrBg87n2CR/sSCdtsZYsXEVY4WLVoZ4z+9FHr/6asivvYbgL3/pfI2iIP7+91ui22y1JhnlCWbPd8tt3jDHlEZG+IKjjfjVVyP+sY/Bf999lut98tJLrXaKlhhbsQL1H/oQjyabPebjcR4VbmnhJnN33onBn/1sXO7cakcHlL17XZepVBPjzS6wp2irzz2Hmq1b+TMx1a3IKs0UMHAbmXecH6PVIVQVzDQ5FL3RBQKBQHCKEaJ9GmBNPA4eBHQddbIMbfHi4imBkgTa0MAFRDjM6w2bm6F0dZWcbDrqRRlzphu7dNt2UGGv86qi0nHbo5qTgZGq6iazgKTTIJSCGm7HY6JpIIODrs3ZMlDRjbl4CFvwF8NErhctZV+z2NOFNnknNicfxPl4Ag0YRAAJqCjMZKB1dWChEMixsaPsAECbmpA55xwEfvvbMbdloRBPNV21aswsCmlkBJHPfIZP8Bcv5v2/T5yAbgpyWws0kkiAeb28V7iigPT3Q3ZxPtXnnuMt3ypEPn7cWdNd4t6ThobAGKvYuKpoL3IT27Nhbudpb4dvxw4u5GTZUf/u27EDqfZ2672loSEeqZWkXOTe9lbmQmHRRRtFQfKyywrHa4t6mrXqzOOBcviw0/Hc6+V17q++CrWzc1zu3Opzz4157zjKVKqIk+r9baRoZ1etgu+RR05JK7LxZApUEpkXvdEFAoFAUO0I0V7l5E88EAqBxWJlUwLzJyAsFILW0gLl2DFHzbqZOl+0JtjoES0b0UQaiTgn0+WE+XQV7OPA0VpqovdtttGy1fCWxfAtkNy608fjZc0IGYAYgmjHJjyMi/E4LsDLOBMUpd3pazCCc7ETm/AkLmp4HotaRhEc7EYw+fqY47H62Ls8Xrm3F/4HHnCxZc6t3X/vva4i4WbbNHXfPr6opWlQ9+8vmoZuthNDNsufFxeou3e72i4f0/zNcu+3i3czY4ZSJP7pn5B697srN67K70Ve7L423fjt0c/ZswtEXH70k9bVAcb4mDluW+o9MYzraF3duM6NNDjITcOSyZzTv7kQmUxCymTAAgFIg4NIX3BBxe7c8vHjrsbhdruyTLDZ2oT0/pYkxD7xCUQ++1nIXV2g4TBYOAySyUx6K7KKMwUqjMyL3ugCgUAgqHaEaK9mik08JKlw4nHOOVA7Ox1povkTEFZfDzY8bKXJM68X+qxZUI4cKZ76LkmAx2NNqguEznQS5pOY3k/DYTCfjy+ITMobUPf1+6YZ2kmUD+iQ8CoW4wG8C3/GW7ETGzGMSMntCShWYTc24QlciEexETsQxih8SEOvWwymBEAS7tLdEYuBmC39XMAq6Ikt9/QA4DXbbiCpFGhDA3/OurqcrvM2kUxDIeitrZD6+yH397t+LiruTmDgMH8zzCSJLXptmr+N3nhj6VZsZQSh1Yu8lPmhrRd5pdFP2tjIa/LNdmnF9h0O8zZi44BGIjnBrqoOkzSmqnxxMpGw3OYrdefWjV7yY6G3lM8+GYtJMVubgN7fnvZ2hL7/ff6aRIKXmJw4ARoOI7ty5aS2Iqs0El5xZN5FpxbRG10gEAgEpxIh2qsYNxMPdc8eNL7nPZB6ehwTvPTmzVCOHHFM0GhtLeREgqfM19VB7u3NCRCztl2WeTRM07jRFmM87XUy68gnGb2xEZDliYmA5SFHo5bL9kTjcNZ2wzhT9UcQwqN4Kx7BxfgrLsABnF52+0b0YSN24C34C96GRzAfR+BHEpIt2ZmpKhcFyaTryL8yOgqMjroeNwsEAJei3RRSbtvbIZvlqe8+H89G0TRop51mPRtMUcC8Xsg9PaCRCOIf/SjCX/86YNRljwUtJuzcRFdNR/Wbb3ZkyFgmh4Qgdu21JQX7WIKQ1tfn+qkXgxDeZq2+vuLoZ3bVKuhz5pT1LNDnzKk8mmmcN/W550qXDdgEvAOX7tye9nYEf/UrV8NhZgu6ceAqpXucbcDG3fubUvi3b0f49tuBdBq0sRH6rFkgo6N8H14vYp/85KT2Dq80Ej6eGn63nVoEAoFAIDgVCNFexYw18UA2CykaBdE06M3NjgmecuQI4ldcAe/jj/MJWjQKxhi0+fMBRQEZGirsWW1zOGeynBMFZpu4aYpstOyqBAa47gPvtsf8ZMMAV5FeCuBlnGlE09+Gp3EOkgiU3F5BFuvwHDbjcbwdD+NN+BtCiENG6XuCUMpdvVWVi5g8c7GThQFWj2g3x2w6XWvLlrnavxSLQUomc23cjPRtGgw66rBpJALl4EFu/idJ/FjHEu2EFNSOVxxdLXXcZRZtXAnCc87hC0WlFouM32VXrIDa2Tn+OmBZdtbAj7MbhOO8xeNWzTlJp3PRdtNc0shSct0pwfYekRtvzHULGIOi6f1uFmRcpHSHb745d8+NIwpfaXaBp70doW3b4Nm1iz8HRmcNvbkZLBKBXlvLM76+9z0Mbtw4eZHoCjMFxluj7qZTi0AgEAgEpwIh2quYshMPxqxIud7UZP2e+f3QvV7I3d3w33svhm+5BepLLyHwi19A7unhfdwNIUXjcUiZTNHJP7G1hCNuU+FNMzugukS+rkOqMMruOl7NWEUCvyLM9nMu3eMBboTFZLlABA2hFg9hCx7GFvwVF+II5pfdTyuO4Hw8jrfiT3gb/oQW9HKHd7dQisRll/HU8RMnULNtm/vXuoCALywxl/en8uqr/HUunfWZYQBIMhkuVI1rUbCdEbEDjOwCYMyFBOb3g9pSrStqZaVpCN1xB3++8t+HEC787rgD8Y99zBltpxShbdtAhoZ45o5Zj59XajMaCo397FIKtbOzdPTTrCHv64O2aBGyK1YAKN3znpjjqK2FNDBQvK1WEdHr2bnTcd6Y3889Goxaf6JpOcM7nw96bS0IY5WZidmENA2HIY+1ICNJ1v1g4nZBxpFZBd773TI99PvBPB6onZ2gNTWgTU3uW54VGaPb7ILIjTfyBV5byQFJJiF3d0NvbQULhcbfn71CKskUOKkadUmCtno1MHs2tOPHp1cpmEAgEAhmLEK0VzHlJh4kmeS9kb1eh6AnsRjk3l6QVApSZyfqP/hBLvq9Xt6D1oxO9PVBmsjWVLLMo0OqylP186P4pxipkr7rhIBKEiQ30T/GJkewA1bmg1WPOwaEUkBRQGUZkq5jF96MB/Au/AlvH7Mdmw9JnI2n8Vb8GVvwIM7Ey/AUcXh3DWMI/PzngJEiPxlIhw+7XlAiySRAqesSCWL0JLcvgDCPp0BImRG77Nq1/Fnt7Bxzks+yWUvIVmqY5b/vvlz6ff77GP+XRkbgv+8+h9t6YPt2eJ55hruzx2KWgZ7e3OwUXs89V7YdHmBkIfT3F68Dzmatzx8QAqWrC/Uf+hBi11xjtZTM73nPFMVqASf19ha01TKjvcrevTnRe/rpkIaHneeNMV7OkEzmHOObm61SDbmnp2IzMbuQHss5HgD/7LBF2itZkDEzq1g2C+X48QL3e2gad9+3tVwbq+XZuLHdlywSAYxMErs/gNzbCy0YnNJ2aK4zBSaghl8gEAgEgmpCiPZqppw5Tl8fr01vabEm+iQWg9zdzaMisgxiRLxINpszNDNSRGlDg7s0UZeiiAYCfMI8PFyd9e+VtKFjzJ1gnyKIy6yFXjThfu1deEjbgsfwFvShuez2y7AfF+CvuBgP4UL8BbVwX0/uCkq58Vg8PikLG3I67bolIQsEUP/BD0Ldu9fVvhngKBcBpVAOHeLCze4aL8vInnUWsqtX82f1uuvG3LeUzUJ98UVk3/Smig2z5O5uZyS8WEs2Svl2Bp72dtTcfrsleM1z5oiYBgKQhochHz3qKtIu9fUBcEY/1T17+GeKaXTZ0gKoqiVQY1ddVbTnvXUoqVRByrKnvR11119v+WuYyE8/DVDqdK0321t2d/N2huk0kE4Dug55aAispqZioeYoUXLz+SFJufFXuCBD6+v5fWZcg3z3e+vzW1Wd71mi5dnJ4LgvzeM23h+EWIaX1gLJVLZDc5kpMO4afoFAIBAIqhAh2qucUuY42qJFvPe6OYFjjKe/G+nRZh06gdGWTNch9/RAN3p+u63PdIs0OgqpAhOxKacaUhwNoeem5ZgJk2Vuilbkdxpk7MR5+CMuwZ/wNnTgLDCUFiRhDGMTnsDb8Ce8E/djMQ6BVLKYUSFSMgmpq6syM71KcTn2wK9+xRcRgsEyZygHyWYBVQXz+cBUlTueGzXulpCKxUAUBenNmwFJQqatDdmlSy2n+nLUfPvbGP3CFyD191fcyio3yCJmazbhbv4duvNOLl5NgzkzZdwWMdVnzQJTFB5Bd4Fy4ID170xbGwbPOQeN73kPr3c2y3WM8ZkC1ffAA9AWL4a6b5+7lGVKEb75Zki9vZYwtM69sRApDQxwsWgK4lCIt6rs6eFi/cQJLohrahC/4oqKhZq9RKngHNsxz6ltQaDSBZnsihX8WdE0Hlm3u99LUi7jo8i9MtHRbnvUX+7r44sgtrGbnQugaZDi8apth1ZpDb9AIBAIBNWKEO3TgKLmOGecgfoPfchKnZcGBnjqrtECykLXeYo1ISDxOJRDh/jPq6nm/I1Epe7uxvUz6cZc/BGX4EG8A4/jAoygtvRbgeIsvIiL8CjeifvRhnao+XXpk7mYYeuVXUldfiW4rWmXBgd56YbLmnYaDILOmQPm9eaeGcM5PrdTCdTrhffxxxG/6iouBFy2cvPs2oW6j36Uu8hT6towK7t6dU44ljpuQqxIpCUcm5q4C34qxReCTJFpREyl/n5kV6/m5n4uyC95UDs7IfX08HT7/OMwBerBgxi99loo3d2uUpbVjg4o+/dbCwwOEWt4PZBUirv8B4P8V7EYX5A0XO71piYwjwdSMong9u3IrlpVkXC3lyjRYJCPzeb3YTnVG+fUbnRXqYO52tnJ9yPLucVXc5HC1h2ApFIF16ms6d84sKL+R44ULrqZiyaSxK9bXV11p5q7jMwLBAKBQFDNVOm3rKAA0xzn4ov534qC2DXXgIVCkLu6eGSp1CTePumb5k7w05pxtGNLazIexttwHbbiDOzFfHTjk7gLf8ClRQV7M3rwj/gF/g//hONowfNYj2/jc3gLHi8U7ABoKAQ6SS3rzF7mkwWVJPdp95LEo6BuvQ0MAUZSKSsF2BKPqmqJSCmTgbpnD9TduwEA2TPPdLV7VlsLFgzycpZ4nEeTi9SoS9EotCVLrCgmbWoCq6kpv2/DqAzICUfYSmRIJsNTm02TPV0HvF7ErrkG2fXrXY0/u3at4/9WZLaMQCWaBjp/PqK33ILsGWeAxOO8/t2I1OYbqakvvOAUr3bsPzNbChrmnFZE2jD8Y5EIb1EWi/Gsg0o+/4wSJRYKcTGeF/E3x8J8PmjNzWCBgCWcC6L0eeQLbWlwkH/Ot7byUiNKuQ8ApbyUwKj7L7iHi9wnJ0t2xQoglSqfJUMpsqtWuTfAEwgEAoFAMG5EpH0ak2lrQ/Sb30TdJz4hhPh0wBBN5dzmGYADWIYH8E48iHeiHW1IoXRvcRUZnI1n8HY8jHfifqzFi+6FrCxDX7oU1O+Hd+fOMTdnHg930Hbbvs3mJu56+woi/xIh7lPvDTd4SJKrZ4UahmaWmDZriQmx6rCZEXGURkastHJ97lx34zEi5XpLC2RDRMvHjoEGAtYYpWSyoA47u2IFr9su5WJOCLQFC3Iiv0idNNF1/v42o8ORf/1XLrw0DbU33ZRz2S9SM89CISTf+17n+aqgxVZ29eoJSVlmRkRaGhkBNVzvrdITWYbe3OyIzo+37tsqUdq2jZv5ZbNcqHs8IHV10AKBokZ3lTqYm+cQqgrttNMKjPrI4CCUEycgRaOghrneZBmrqbt3884iJsW8E2QZozfcgOy6dRPyngKBQCAQCEojRPs0h0UiYIEAdI+Ht4ATVDUkm0W+LB1BDf6KC/FHw+m9CwvKTfYC6QAANiZJREFU7mMBXjPq0h/AW/FnhBAf11ioLEM6cgSpyy93J9pVFXTWLJBMJudg7ga3Zoa1tTx6PTLiSowzvz9n0DXWtubfsuzK2E9Kp8HiceizZ0OKxYpHe81aY0ohDQ0B4CndbpAGBrgxltfLj5sxIJOBYhi5gZCCOmzLRf311wvGYRny5Y2xWJ00U9WcyZ6mgQWDSF5+OX+BomD0059G+Jvf5K/Lv3ayjNFPf9rZTg7jaLHlImU5u3YtXwgwo+15EMa4wF261IrYw4hK6y0tYKGQY/uTqfs2S5QC27ej5vbbgXQatKYGkscDpFLFje4qdDB3nMNZs5xGfYxBymSQXbHCKjWYTGM19YUXrPIWYrTQs+4vI2uFUAr1pZeEaBcIBAKBYAoQon2aY6WlFpnUCqoQxsBA8ALW4CFswYN4J57B2dCglnxJAHG04UlswYO4BPdjCQ5OyFCkTAYYGEDwnntcj52k02DhMOBCtDMzcu42CySdhr54MfQ5c6Du2TP2/r1e0EDA1WKV1Wfa5VjSmzcjfs01kPr6UP/Rj/KIdLH9Uspr2402X8RM1R5zQIbgTyYhp1JcbIZC0JqaIFHKWw5qmlWHDcDqmQ3GLNEEgJ9jw9lbr6tz9DovWSdtvs74ndrZaYno+NVXAwBCd9zBF1DMRYRwGLFrr7V+72ASWmxlV6+GtmwZ1M5OkEzGaURnpIhnTz8dA/fdB7WzE+pzz6Fm61arZ3vBKT/Zum9JQuLDHwZJJBC64w4ox44BjEEpssBiUpGDuYtzOHLTTchMpbGamWFCKQhj/Jk26/pFdpdAIBAIBFOGEO3THCul0kUfb8Gpow+NeARvx4N4B/6Et4/Zjm0ldlsp723YAS8mx8gNjLmOmlOfD1JfHxdPLiC6zkWW28m9sSgAN33dCQEzje7cIEl8ccvlWMw0brWjg7etGx7OLY6ZwtE4PhoOgzY2AgCyK1cCv/mNq/GYgshK6c5koCQSPDVeknhkPJtFaNs2fshmz+yRkeLZApRa4s6MJtvrpOX+fmfvb58PemMjpESiIPocv/pqxD/2Mfjvuw/y0aPQ585F8tJLCyLsdioSqJS66rU9ctNNiFx/PeSBgVxavyEc9YYGjNx0E6Ao3H191Sr4HnnEfbR/HHja2xHcvh1QFGhz50JVVWjZbFmju0oczN2ew8k2VivIcpAkZ3s+YxEs39tAIBAIBALB5CBE+3TGMJWj9fWQX331VI9GYEODjKdxDh7CFjyEd+AFrC3bji2CIVyIR/Eu3I+L8TDm4PjUDJQQXpvrYtFH0jTe1s9turssg7W0AMPDkF24ttNwmItON1FDxsBqa123LtRrayFlMrla7TEgsRjUjg5kV6xAduVKqC+9xN3XMxn+3EkSmMfDhcvKlZYQZEbEfczhF6n3d7QCNE3IZBlqRwcvTairGzNzgWgaEIvxNmNwUSddpDe6haIgedllro7HxI1A9bS3W6LU7B2vLVlSNMU709aG6NatCN1xBzf7y2QAjwfZVasQu/ba0pHq48dB/f6y/gAVY++7bvaHV1WwbBZ6bW1B33UHFTiYV0ObMjdZDtqyZcKVXSAQCASCKUKI9mmKfeIrDQ9X1PtbMDl0YR4exsV4CO/Ao7iobDs2CTrehGcNSf8g3oy/QcIpSDetIBJeiWAHwFOvGXPVkg0AUu94B/xPPAFy4oSr7ZPvfCeCv/qVu7F4PNAjEUipFCQXDvL+Rx6B9+mnoS1ZgvTmzVAPHLBqpi0Tt0wGLBx2CEHqUrTzjenYiyW6Dml0lLv8e70FrdaKQWypy2PWSU9A9LmAMgLV097O0/yNhQVqpH+re/cicuONpZ3IDYEMo469VCeGTFsb4ldc4UhfL+YPMB4q7bt+UpzqNmVusxyqtc2bQCAQCAQzDPGNO93QNIS+8Q3UffSjUHft4lHJIgZUgsknCR8extvxadyGFejEQnTh4/gf3Iu/LyrY5+AoPoL/xa/wPvSiCU/jXHwFX8XZ2AX5VAh2wOm4PRYV9nTPLl2K0Rtv5GnVLkj9/d8jesstVqr5mPj9GPn0p11tKvf0QHntNVeCHQCYzwcWDELduxfB//kfIBbL1fGa0W5dB/LaeZmGdGNhtfLKTzc3Wss5nmdKeT96t6UDANSXXjIGlGtZJp84wUU/pbyW/sSJCXcdL4s9Um0sIECSuHFciZZspshX9+0DjUSgz5sHGolA3bcPkRtvhKe93fEW+enr+rx50ObOBVQVwe3bC7avBLdt7cZjdFeNmFkO6XPOAa2r4wtHdXVIn3MOolu3ijZvAoFAIBBMISLSPo0I3HUX8N//jdDoqPUzqbsbgOGK7bb9lWBcMAD7sdxIed+CJ7C5bDs2D9KGgdxDuBgPYxV2u2/HNpVUKMbdonZ1Qfna1yC5XBRQXnsNyfe9D9qyZVCOHBlze097O0a/+EUeeR0rYm0eYwVt5ZjfD93rhbp3LxfosswN4MDvBUIppOFhhG++Gf0PPMDr5qNRV/umkQhfnBgdhWJmFhSJ3lpt1hobeenAGDX85vjsVFRrPolUHKnOE/nma5jfD93nK0xHL5K+bl7pMdPXXVBJW7uZQjWk6gsEAoFAIBCifdoQvOsuhL/xjZKpzEKwTw7DCOMvuBAPYQsexpYx27EtwSvYggexBQ9hMx5HEC7dxGcg0ugojwy7zAKRjx4FAKseeyw8L7yA2htvBPV4IJmR7wnCFHsklbL2ywyn9fw+7cr+/bz+fc0ayC5T+826fcm2AFeQMWNbaEidfz78jz3G3ePLYPZdzzcIG5f4cmMWVwFmpJqWiVTbW7JVKvInO3294rZ2M4VTnaovEAgEAoFAiPZpgaYh9N3vihY7UwAFwfNYh4dxMR7GFjyFc6GXeUxCGMVb8Bcrmr4Ir03haKscQngGSMad8710nJvvJT70IQR++9uxd5/NQt23D2AMen09pHjcWfMtSTyt3BTzFUTZTZf0si3czOPTNKgvvIDsmjWun1EyOgp4vdCWLoVnz57cgkNeP2wwBqYo0E87DaPLlsH/xz/Cu3On8zjyTe283uLCsQLxVYlZnFsqjVSPKfI9Hkj9/fA+9hjfvr+/okWBiinSkg2yDJJMjrutnUAgEAgEAoEbhGifBvjvu88ZkRNMKD1oxiN4Ox7BxXjERTu2NXgBFxsi/TzshAei3V5RjLZmVl/nMaAtLfwflfgzGPuVh4agzZkD5ehRqwc6JMn1gkE+RdPQXYxL6utztf/M+vWIfelLyK5YgcZLLuH91I33tdLvNY0b+QGoueMO7iDf0gIaCEAyatP5i2yp/0arOHvf9UoZt1ncGFQaqXaIfJ+PL8gYZQrQdcg9PSDpNII//CECP/sZv38ondT0dUepwcGDwOgoiCxPeamBQCAQCASCNxZCtE8D5KNHRZR9AslCwU6cZ0XTX8C6sts3oB9vw5+wBQ/i7XgEs9AzRSOd5oyzVt734IOVvYAQQNcddfBWdN00aqwUQ7QXiD/TiM4wi8vvVy31uLw3CLFEtcOl29g/Mc3uJAlaSwtYJAKSTkM+dAhSMgm9qYlnFqRSrvqum2MfM9290jrySqiwJZsl8l96CUTXcz3mzXPDGB/XnDkgmQzk7m6QeBySpkGfP3/S0tetUoM9e9AEYAhAduVKEWEXCAQCgUAwaQjRPg3Q5861JreC8XEYCyyR/iguwijCJbeVoONcPIW342FswUNYh+dPnbv7NIbZ/nYTO6ehEABANswVXVMu7Z2x3LNjiu2xRLxRu06SSW7aZvSxJ3lO8QC4qLb3q3b7jNq2s3qRb9vG0/3TaR5Jl2Vora1gNTX8UPx+0KYmSKOjkEZHed/1kRFImQyoxwNWW5uLJkciUDs6LIFOolGEvv/9MdPdJ7suvKKWbJKE9ObNvBxA13n2g1luYVxDM0XdWlTIZPjixvHjoHV13NE9nZ749HVJgrZ6NTB7NrTjxyfNzFEgEAgEAoEAEKJ9WpC89FKEP/95SPH4qR7KtCEBPx7HZkuo78fpZbdvxREr5f0iPIo6RKdmoDMZe9q2CzzPP48EAH3OnMrep5xQzjN3Y5JkRbOLIklgsgxpZIRH0FesgDZ/PoK/+EXxFH9CkHzPeywhOGbfdfNledvZjeLU555DzdatoJFIQaSf+f1ciCaTUA4dslLoJULAhobAZBn6/PmoueUWKAcP8vehFCQeB/N6QZuby6a7V2oWZ+HStC6/JRshBIwxSMkkgtu3I7tqVU64Uwrv44+DBoO8RV4mAxjHCwCQZd7DvrHRynygzc2QBgehzZsHuafnlDnlCwQCgUAgEEwkQrRPByQJtKUF0qFDp3okVQsD0IkVlkh/AucjDV/J7b1I4Xw8YQn1FeisznZs0xjCGFg5gZyHd8cOeNrboS1bNrEDoRTarFmQYzFeF11uPIwh8eEPI/2Wt3DxuWIF6j/0IdBQiAtHM0WbEDCvF0xR4H38ccSvuipXv++GYtsZRnGmIC7aD5wQ0HAYcjIJkkpZ0WdTmBNCgO5uyyiNejxc3KfTud7yZm/0Iunu42lr5tq0rsKWbFbUv7mZ17SnUjzLoK8PTFV5GUE6DZJMggUCuXMmy4h95jOgjY3Tt03ZBDv3CwQCgUAgmN4I0T4NUHfvBkmlQL1eyMVSdN+gRFGLP+OtllDvxryy2y/HPkukb8bjCCBZdnvBycFg1Ja7bcVmiLrs6eWzIsZD6j3vwejnPw//vfci/NWvQirVOo0xeB99FCNf/jIXjh0dUF55BTQctgzQiCyDqaolJO3p4vq8ecCzz445Hn1e6Xu1rHBmjJtSGosG1vklBMzv56ngsRiyy5dzI75kEkTT+P50HXJvL7RgMBeZzkt3r9QsrhLTukpT7x1Rf+P4KABpYMBZ6mC7v6xFhcbGadumbDKc+wUCgUAgEExvhGifBkiDgyDRKKQ3uGDXIeE5rLdE+jM4u2w7thqM4CI8agn1hXh9CkcrYJIEVlcHGgpBeW3sVng0FILy6qtgE9hv3YQkEjwle/Hi0oLdQHntNagvvojsunXwPPkk5P5+/gtbhF1vbrb+bU8Xz5x3HgL/7/+NOZ7MeeeV/F054UzMCLvPB23JEpBUiotyRQEYg3L4sBWBZn6/lT4PWebt74pEph3p7pWYxVVoWldp6n2xxQvm8/HFilQKTJb5e8qydX2me6/0yXLuFwgEAoFAML0Ron0aIL/+OqRy/aJnMCfQgodxsdWObQCNZbdfh+cskX4unoIKbYpGKsiH1tUhevfdUJ95BrVf+cqY25NslrumT8ZYGvl947v/flfb++6/HySRQPAnP+EmaLLMHeUZA0kmIXd3Q29t5WLYli5etq+7jbLbFekHbhmq9fUBhEBvabGiz+YZI2ZbSMa4kIfRus6MSJeLTNvS3d2axVUaOa809b7o4gUh0JuboRw5ApLJOGr8p32v9Ml07hcIBAKBQDCtEd/81Q6lCG3bdqpHMWVkoOIxbMaN+BbW4XnMwQn8M36KX+DyooK9EX34ILbjblyB42jB3/AmfAM34Xw8KQT7KcZMUZaPHXP3gkQCTFGgrVgx8WMxHNjdOtPLR45wAWUIQ2KKXklypJpLQ0PQlizJRXYlaWzjPWM/5TD7gWfPOAMkHofc2wsSj0NbtIiLZFUtMmjZ2r/ZZ96KTOt6TrjnRaYd40ehWZw+bx60uXMBVUVw+3Z42tv5oRqR86K19+CRc6JpVuTcFOFSNFroK1BsLMbiBQuFIJ84wf0IKAVkGTQQ4J4Cfj/kvj6QeBzZM86Y1pHoShZBBAKBQCAQvLEQkfYqJ/zVr7oXPdOUQzjNSnn/Cy5EDDUlt5Wh4TzstNqxrcULkCDaLVUlhPCa8AMH3G2eSCC7dq3V4mwikXt7Abh3pmceDxdQdXVAOAylu5sLVCMlm0kSSCIBGg47e4uvXWu1iAPgdLY3trH3dS+H3VHeMiQzjPGKpc4z2/8tIV1pZLoCs7iKTevKZRCUiJKbixdmjbflBn/WWYh98pNgtbUzxqxt3M79AoFAIBAIZjxCtFczmobAL35xqkcx4cQRwGO4wBLqr6C8W/gCHLZE+oX4C2oxMkUjFZwM6oEDqPvAB0BSKdeviV1zDXy///2Ej0WfPRsAXEfxaXNzTkBJErTWVh7tNt3jAUCWEf+nf3JEdrOrV0NbtgxqZycALv7NtmbESEt39HUfC8NR3k454UsbGsAAyD091u/MyLQpruW+vpJt0CpKea/QtA4oI8LLtGQrungxzQV6Mcbj3C8QCAQCgeCNgRDtVYz/vvtc18hWMwzAHqzEQ9iCh7EFT2ITMigeTQIAH5K4AI9ZtenLsV+0Y5uGEEohR6PutzfStic8s0SWkV2/HgBAGxosF/hy2+tz5zoEFAuFoAWDOeM3XQfRNGTOP9/5WknCyE03IXL99ZD7+620dGLWYzc2YuSmm9wLTrP1V38/pKEh0Lo60MZGRL/5TYS+//2iwhfAuCPTFUV7xxE5B8YpwossXsw0xrMIIhAIBAKB4I2BEO1VjHz0qOse19XGECL4M96Kh7AFj+BiHEVr2e1XYI8VTd+EJ+GH++isYIagqgjdeSdYBZF5NzBVtYQObWwEjUS4g7w9dd1EkkAjEWTXry9qgsb8fjDGIJ84UVJAZdraEL/ySoTuuAPSyIj1DNOaGsSvvNJ1zbXZ+kvds4fvh1I+vnAY2ZUry4rw8UamK432jidybp7nmS7CK2aciyACgUAgEAhmPkK0VzH63Lk5x+cqR4eEv+FNlkh/BmeDQi65fRjDeCv+bEXT5+PIFI5WMGVUcP/SQKCilm8M4BHsMfZPdJ2nc69Zg+yqVciuXAn1pZe4W30qlWvl5vNxgb9yJbKrV49bQDmM3FpboaoqtGwWUjKJ4PbtyK5aNaZwt1p/DQ7yzhGU8jp6SiEND0N96SVEPv95RG+5BekLLijcwThF8XhT3t8I6etTwbgXQQQCgUAgEMxohGivYpKXXorwdddVrcX/Mcy22rH9CW/DIBpKbktAsd7Wju1sPCPc3QVOKM31HHcBU1XQ2tpcH/V8AS9JXOjqOtQXXkB2zRormll3/fW89MQUpYRwozmbsVxFAsqWxh667TaQ0VHos2bl6vk9HujhMOSenrHbdplmcKOjPL2eUjBVtUzhSDbL+6/HYhPfAmy80V4ROZ8wxCKIQCAQCASCfIRor2Y6OqpKsKfhwQ5stKLpHTir7PbN6MHFeBgX4yG8DX9CE/qnaKSC6YjZPkxbsID3Bx8DqiiQkkkARj9yVeXCnzEws62aIXrzsaS9TbQ7fm6QaWvD4DnnwH/ffZCPHoU+dy6Sl17Ke7YbmGnsyquvclf2kREwRYESj3NxzRgUQsC8XtDaWkfv8mJYZnB+PxRjocDhEi/LIJkM9IaGMfc1HkS0twoQiyACgUAgEAhsCNFexTT93d+d6iHgVSy2XN7/ircgjlDJbRVksRE78HbjFWvwomjHJnBPNgttyRKkN2+G76mnxtxckiRoS5fCs2cPCKX8TpOcdxzRdWeLNTOKrevILl8Okk5b0X3m9RZEwh2CPJsFU1X4f/MbS7xaaeyxWK5/+vAwd5kHX0wgqgpGKRf0mQxYIFC2bZfV/1xVc73V7RgZBYQQRx/0iUREe08BZraGON8CgUAgEAjyqErR/tBDD+EPf/gDotEoFixYgH/5l3/BkiVLSm7/1FNP4Ve/+hX6+vowa9YsfPCDH8S6deumcMSTw6mYrsUQxF/xFkuoH0Tp8w4AC/EaLsZD2IKH8Bb8FWGMTtFIBTMOI/Xa+9e/uto8tWULhm+7DY2XXAK1s5P3ITej0ozxKDecLdYcLc0kiRvL2fZpb2lGRkYcgpwaaeLq3r2I3Hgjot/6FkLf+x7vaT5rVi4935aiTyjlP5ckbvCWzQKJBH//EphmcDBfmy/czRp8xoDJbAEmor1TRrHFIW3JEpHZIBAIBAKBAMCp0YVl2blzJ+6++278wz/8A2699VYsWLAA3/jGNzA8PFx0+/3792Pr1q248MILceutt+LNb34zvv3tb6Orq2uKRz7xTEWMmgHowJn4T/w73oo/owGDuBR/wPfwqaKC3Y8E3on7cTuuwz4sw0EswvfxSVyK3wvBLiikAhPF+D/+Y0UChTY1AYqCkZtugt7UBBj16ySb5bXgkgS9qcnRYs2KYpdpaUY0jdem33mnJciZ32+JfH3WLJBYDOFbb4XyyivFe5oDOcFtpufnpeKXwjSDk5JJazyOhQBdB/N4ICWT0JYsES3Apjlmtoa6dy9YMAi9pQUsGLQWhzzt7ad6iAKBQCAQCE4xVSfa//jHP+Kiiy7CW97yFrS2tuLKK6+Ex+PBX0tE3x544AGsWbMG7373u9Ha2or3v//9WLRoER566KEpHvnEM1mifQD1+CX+Ef+C/0UrjmINOnAj/hN/wUXIwlOw/Sq8jM/gO3gEb8UA6vFHXILrcAeW4RXRP30mY7Q5A8AFq8djmbu5RpLcpfiqKrJnnw0A0OfMcbVrc7tMWxuiW7cifc45vI95KARaV4f0OecgunWrYyHA3tKsGGZLM2loKBeRL5KeTiMRyF1dIIawzg1Kzx2vGXU3hDvJZi3hL5XrX2+YwbGaGjBZ5osR2SygaSCZDN+1oogWYDMBs1yjzOJQ6M47i7cnFAgEAoFA8IahqtLjNU3DoUOH8J73vMf6mSRJOPPMM3HgwIGirzlw4AAuueQSx8/OOussPPvss0W3z2azyGaz1v8JIfAb/YjJGBGwqWaipuIaZDyLN1sGcruwAazM3iMYcrRja8XRCRqJYFqhKFwsGGnaJJsF83ignXEG1I6O8kJCkjD85S+D1deDRKOo+c53IMXjJTentbWgTU0ghCD7pjfx2nDbc1qAqiL7pjdZz2x20yYMbdwIxVYTrBk1wfanWjvzTOhLlkAp09JMO+MM0Pp6EE0DLRORB6WALDt7mhsiG4RwAc8YF9uSBObzgdbW8uNtaCj7eZPdtAnDt96K0LZtUDo7IQ0P81R7SQKtrYW2YgVin/oUsm1tYuHsFGFev5P53lD27IFy8GDZxSHl4EGoe/ZAE6UKU8ZEXFtB9SGu68xFXNuZi7i2OapKtI+MjIBSikhevWckEsGxEm7S0WgUtcZE2KS2thbREpGse++9F7/5zW+s/5922mm49dZb0dTUdFJjn0rc3LbdmGvUpfN2bFHUldkfxZvxLLYYIn0DdkGBu17Zgoljwj6O/H4gHAZ6ek5uP4zxfdXXA/E44PWCfPGL8HzsY8A55wDPPVf6tWvXImKmpVMKPPEE8Le/cSGeyfCfSRLg8QCqCnntWjS97W38Zy0twOrVwAsvFF8YkCRg9Wo0XXxxYZR57tyxj+vLXwY+/nHIvb382Hw+IJUCBgeBSATyl78Mb20t4PNBphQoJtwTCSAUAubMgXz4MFBTw0VXOMz3l0zysfl8wOzZIKoK4vdDOnqUj9081nK8733AZZfx89DbCwwMAA0NkJubIa9dC6+IsFcFs2bNGv+LOzr44k4oVPx+kGVgdBRNADB79vjfRzAuTuraCqoWcV1nLuLazlzEta0y0T4VvPe973VE5s2Vm76+PmhadfUNb0ThBSIonjafghdPYpPRYO0d6MTKsvueheNWO7a34s9oxIDj98LzfWrJv65UlqGddx6yy5fD/4c/gAwNWanRxaCqCjZ3LpKXXILY5z4HKArqL7kEnuefL9iWeb08akuplX4NTQPRdegNDUhdeik8u3ZBPnGCG57pOvSVKxH71Kd4qnlfHzw33IDIdddBGhgAzJprQrgxWkMDojfcgIxt0cBz5ZWoffVVbupWXw9iGKlJySRYKIThK690bm/fP6W5/UtS0f1XxBlnwPPNbyK0bRvkgwdBBgbAVBX68uX8GM84A6AU9QsX8oi8aTJnnUAGub8f2hlnIPbJT6L2858HOXLE6mkuhcOQ43GAEOiRCJSaGmRjMUhHjhQ91jGZM4f/sXOyCzKCk4YQglmzZuHEiRPcFHAcKADqZBksFstla9jfI5kEkWUMAdCOHz+5AQtcMxHXVlB9iOs6cxHXduYy06+toiiuA8dVJdrD4TAkSSqIkkej0YLou0kkEikwqRseHi65vaqqUFW16O+q7Wbo++1vMfuyy4r+jgF4BUvxELbgYWzBY7gASQRK7ktFBhuxw0p5PwsvibTaicIQq2XTufNgNTWgPh/kvr6C32mzZyP6v/9rtXxKve1tCN15Jzx/+xtIIuHcWJaR2rwZsX//d2eLKMYw8Ic/AIkEItddB6WryxLh+uzZIPE45N5eXtudzQKMgdbWInrHHcicf37p9lPGM5LeuBFDW7citG0b1H37ePTc40H29NO58N240WGelt640dH7m2gaYO/9XWT7SvZfKemNG5E+99zSx0gIRq+5BpEbb4R84oQlyEk6DSkaBQuFMGo4exftab6SL5pJg4PA8eMgslzyWAXTG8bYuL87sitXQlu8GGqZco3sGWfw+0ncM1POyVxbQfUiruvMRVzbmYu4tlUm2hVFwaJFi7B7925s2LABAEApxe7du7Fly5air1m2bBlefvllvOtd77J+1tHRgaVLl07JmCeVc84BRa62fRQh/BUXGg3WtuAwTiv78tNw0Ep5fwv+ihrEJn3IMwJZzv1bL14moDc2IvGRj0CfPx96YyNCd90Fdc8eSCMjzsizKQLNNl2BADJr1+ai1rEY6v7lX+A/cQKpWbMw9L//y1NlbTh6Zvf0wPvnP4Ok09AXLOBGZJ5C80CLQADRH/0IQM6l2hSh2oIFIKOjXGgGg4jedhsymzbx17lo91VpL+/J3r5ixjjGkoLcFN+GwV3JcQJQ9+xBE4AhcIEmTOMEDgzTwXKLQ8JsUCAQCAQCAWFVtmyxc+dO3HnnnbjyyiuxZMkSPPDAA3jqqafw3//934hEIti2bRvq6+tx+eWXA+At377yla/g8ssvx7p167Bjxw7ce++9uPXWWzF//nzX79vX1+cwqKsmeudejE/ju2hHGzQUzxIAgADiuAB/NST9Q1iCg1M4ynFi70Nd5lZkqgra1MQNuRKJ3LZmzXQemTVrkGlrg/exxyAfPcrduynlfblVFVJfH4imgUkS6Jw5SL773aC1tQjdeSek0VFrTLSmBrGrr4b60ktQjhyBNm8eot/9LhDIy2owI9P9/ZCGhriLudE/WxoczP2ssbFAeBJCMHv2bBw/fnzSVxEd/aA1DUxRRD/osSiVdeCCqby2gqllIq+teC6rC/HczkzEdZ25iGs7c5np11ZV1emZHg8A5513HkZGRvDrX/8a0WgUCxcuxH/8x39Y6e79/f0OB8Hly5fjuuuuwy9/+Uv84he/wOzZs/Hv//7vFQn2aoc+9yc8tr64AcNKvIwteBBvx8NoQzu8yNU9U9vfdonBjD/mz6jh1C0ZP9eWLoU8PAxEo5AkCdTng3bmmciuXQt11y4oR44AiQTYrFnQ5s/nDsfd3WCMQZ83jzuLqyr02bN5e6uhIciPPw7ZNp4+AOqDDzpFruGULL/+OtRnngEhBDQcRvJ97wOdPZuLJUrhv+8+yEePQp87F8lLLwU0DeEvfxnK669DW7AAI1/9KjcAAzB6ww3FI6AlRFj8qqsK96+4eExcRKargUmPXs9Epsm1FUxfxHMpEAgEAoGgHFUXaT9VVHOkHQDe+tYm7N2rIhIBNm1K4i1vSeH889OYPVv0753uzPRVxDcy4trOXMS1nbmIazszEdd15iKu7cxlpl/baR1pFxTnS18aQSjE8I53NKK3Nzojb1yBQCAQCAQCgUAgEDgRon2acP75aRBCHB5pAoFAIBAIBAKBQCCY2YiCOYFAIBAIBAKBQCAQCKoUIdoFAoFAIBAIBAKBQCCoUoRoFwgEAoFAIBAIBAKBoEoRol0gEAgEAoFAIBAIBIIqRYh2gUAgEAgEAoFAIBAIqhQh2gUCgUAgEAgEAoFAIKhShGgXCAQCgUAgEAgEAoGgShGiXSAQCAQCgUAgEAgEgipFiHaBQCAQCAQCgUAgEAiqFCHaBQKBQCAQCAQCgUAgqFKEaBcIBAKBQCAQCAQCgaBKEaJdIBAIBAKBQCAQCASCKkWIdoFAIBAIBAKBQCAQCKoUIdoFAoFAIBAIBAKBQCCoUoRoFwgEAoFAIBAIBAKBoEoRol0gEAgEAoFAIBAIBIIqRYh2gUAgEAgEAoFAIBAIqhQh2gUCgUAgEAgEAoFAIKhShGgXCAQCgUAgEAgEAoGgShGiXSAQCAQCgUAgEAgEgipFiHaBQCAQCAQCgUAgEAiqFCHaBQKBQCAQCAQCgUAgqFKEaBcIBAKBQCAQCAQCgaBKEaJdIBAIBAKBQCAQCASCKkWIdoFAIBAIBAKBQCAQCKoU5VQPoFpQlOlzKqbTWAXuEdd15iKu7cxFXNuZi7i2MxNxXWcu4trOXGbqta3kuAhjjE3iWAQCgUAgEAgEAoFAIBCME5EeP41IJpO44YYbkEwmT/VQBBOIuK4zF3FtZy7i2s5cxLWdmYjrOnMR13bmIq5tDiHapxGMMbz22msQyREzC3FdZy7i2s5cxLWduYhrOzMR13XmIq7tzEVc2xxCtAsEAoFAIBAIBAKBQFClCNEuEAgEAoFAIBAIBAJBlSJE+zRCVVX8wz/8A1RVPdVDEUwg4rrOXMS1nbmIaztzEdd2ZiKu68xFXNuZi7i2OYR7vEAgEAgEAoFAIBAIBFWKiLQLBAKBQCAQCAQCgUBQpQjRLhAIBAKBQCAQCAQCQZUiRLtAIBAIBAKBQCAQCARVihDtAoFAIBAIBAKBQCAQVCnKqR6AIMdDDz2EP/zhD4hGo1iwYAH+5V/+BUuWLCm5/VNPPYVf/epX6Ovrw6xZs/DBD34Q69atm8IRC9xSybV97LHH8L3vfc/xM1VVcc8990zFUAUV0NnZid///vd47bXXMDQ0hH/7t3/Dhg0byr5mz549uPvuu3HkyBE0NDTgsssuwwUXXDA1Axa4otLrumfPHnz1q18t+PkPf/hDRCKRSRypoFLuvfde7Nq1C0ePHoXH48GyZctwxRVXYM6cOWVfJ75vq5vxXFfxXTs9eOSRR/DII4+gr68PANDa2op/+Id/wNq1a0u+Rjyv04NKr+0b/ZkVor1K2LlzJ+6++25ceeWVWLp0Ke6//3584xvfwO23347a2tqC7ffv34+tW7fi8ssvx7p169De3o5vf/vbuPXWWzF//vxTcASCUlR6bQHA7/dj69atUzxSQaWk02ksXLgQF154Ib7zne+MuX1vby9uueUWvO1tb8O1116L3bt346677kIkEsGaNWsmf8ACV1R6XU1uv/12BAIB6//hcHgyhic4CTo7O3HxxRdj8eLF0HUdv/jFL3DzzTfjtttug8/nK/oa8X1b/YznugLiu3Y6UF9fj8svvxyzZ88GYwyPP/44/vM//xP/+Z//iXnz5hVsL57X6UOl1xZ4Yz+zIj2+SvjjH/+Iiy66CG95y1vQ2tqKK6+8Eh6PB3/961+Lbv/AAw9gzZo1ePe7343W1la8//3vx6JFi/DQQw9N8cgFY1HptQUAQggikYjjj6D6WLt2Ld7//vePGV03eeSRR9Dc3IwPf/jDaG1txZYtW3DOOefg/vvvn+SRCiqh0utqUltb63hmJUl8xVYbX/jCF3DBBRdg3rx5WLhwIa655hr09/fj0KFDJV8jvm+rn/FcV0B8104H3vSmN2HdunWYPXs25syZgw984APw+Xx45ZVXim4vntfpQ6XXFnhjP7Mi0l4FaJqGQ4cO4T3veY/1M0mScOaZZ+LAgQNFX3PgwAFccskljp+dddZZePbZZydzqIIKGc+1BYBUKoVPfvKTYIzhtNNOwwc+8IGSq46C6cMrr7yCM8880/Gzs846Cz/5yU9OzYAEE8rnPvc5ZLNZzJs3D+973/tw+umnn+ohCcYgkUgAAEKhUMltxPft9MPNdQXEd+10g1KKp556Cul0GsuWLSu6jXhepyduri3wxn5mhWivAkZGRkApLVgtikQiOHbsWNHXRKPRgtTq2tpaRKPRSRqlYDyM59rOmTMHn/jEJ7BgwQIkEgn8/ve/x0033YTbbrsNDQ0NUzBqwWRR6rlNJpPIZDLweDynaGSCk6Gurg5XXnklFi9ejGw2i0cffRRf/epX8Y1vfAOLFi061cMTlIBSip/85CdYvnx52bRZ8X07vXB7XcV37fShq6sLX/jCF5DNZuHz+fBv//ZvaG1tLbqteF6nF5Vc2zf6MytEu0BQZSxbtsyxyrhs2TJ8+tOfxp/+9Ce8//3vP4UjEwgExZgzZ47D8Gr58uXo6enB/fffj2uvvfYUjkxQjh//+Mc4cuQIvva1r53qoQgmELfXVXzXTh/mzJmDb3/720gkEnj66adx55134qtf/WpJcSeYPlRybd/oz6wouKsCwuEwJEkqWAWMRqMlazUikQiGh4cdPxseHn5D1XZMB8ZzbfNRFAWnnXYaTpw4MfEDFEwppZ5bv98vouwzjCVLlohntor58Y9/jOeffx5f/vKXx4zQiO/b6UMl1zUf8V1bvSiKglmzZmHRokW4/PLLsXDhQjzwwANFtxXP6/Sikmtb7LVvpGdWiPYqQFEULFq0CLt377Z+RinF7t27S9Z1LFu2DC+//LLjZx0dHVi6dOmkjlVQGeO5tvlQStHV1YW6urrJGqZgili6dGnR59btvSCYPhw+fFg8s1UIYww//vGPsWvXLnzpS19Cc3PzmK8R37fVz3iuaz7iu3b6QClFNpst+jvxvE5vyl3bYtu+kZ5ZIdqrhEsuuQSPPvooHnvsMXR3d+NHP/oR0um01b9527Zt+PnPf25t/853vhMvvfQS/vCHP+Do0aP49a9/jYMHD2LLli2n6AgEpaj02v7mN7/BSy+9hJ6eHhw6dAjf/e530dfXh4suuugUHYGgFKlUCocPH8bhw4cB8JZuhw8fRn9/PwDg5z//ObZt22Zt//a3vx29vb3Yvn07jh49iocffhhPPfUU3vWud52K4QtKUOl1vf/++/Hss8/ixIkT6Orqwk9+8hPs3r0bF1988akYvqAMP/7xj/Hkk0/i+uuvh9/vRzQaRTQaRSaTsbYR37fTj/FcV/FdOz34+c9/js7OTvT29qKrq8v6/6ZNmwCI53U6U+m1faM/s6KmvUo477zzMDIygl//+teIRqNYuHAh/uM//sNK5+nv7wchxNp++fLluO666/DLX/4Sv/jFLzB79mz8+7//u+hBWYVUem1jsRh+8IMfIBqNIhgMYtGiRbj55ptF7VYVcvDgQXz1q1+1/n/33XcDADZv3oxrrrkGQ0NDltADgObmZtx444346U9/igf+//buNrSpu43j+C99XKuG4mrW+UBqoVWJD63iOqhtRZ3tfHohGpwVGQyh1ur6QoQZN7YxHNptTtDiNmWszKmnRVSw9QkUh7gi2HRiHSvGWkUllnZWU7vandwvxGDu6O50d13T+v1AXpxz/rn+VxNKuM51/ufU1OjVV19VcXExz2iPML39Xnt6elRZWam2tjbFx8fLbrfrww8/1MSJE//13PH3Tpw4IUn6+OOPg/aXlJQETqTyezvw/JPvld/ageHevXvauXOn2tvblZiYKLvdLpfLpcmTJ0vi/3Ug6+13+7L/z1r8fr+/v5MAAAAAAAChuDweAAAAAIAIRdEOAAAAAECEomgHAAAAACBCUbQDAAAAABChKNoBAAAAAIhQFO0AAAAAAEQoinYAAAAAACJUTH8nAAAA+o5hGKqurtbu3btltVpf+DyGYbywOQAA6E+NjY06cuSIrl27pvb2dq1fv15vvPFGr2K43W5VVVXpxo0bio2N1YQJE7Ry5UrZbLawY9BpBwAAEamtrU2GYai5ubm/UwEAvIT+/PNPpaam6r333vtH7/d6vSovL5fD4dDWrVvlcrl0//59ffnll72KQ6cdAABEpPb2dlVXV8tmsyk1NbW/0wEAvGSysrKUlZX13OOPHj3Svn37dO7cOXV2dmrMmDEqKiqSw+GQJHk8HpmmqWXLlikq6nG/fOHChSovL1dPT49iYsIrx+m0AwAAAADQS3v27FFTU5PKyspUXl6uN998U5s3b9bt27clSWlpabJYLDpz5oxM01RnZ6fOnj2rSZMmhV2wS3TaAQAYlO7fv6/du3eroaFB0dHRys3NVVFRkeLi4uT1elVaWqqSkhLNnDkz6H1Op1NLliyR0+kM7Pvtt9/0ww8/qKWlRcOHD9eiRYueOWd3d7d+/PFHnTt3To8ePZLD4dCqVatUXFwcErOtrU379+9XfX29fD6fUlJStGDBAs2aNUuSdPnyZX3yySeSpIqKClVUVEjSM3MGAODf1traqjNnzqiiokLDhw+XJC1atEgNDQ06ffq0li9fLpvNpk2bNmnbtm369ttvZZqmMjIy9MEHH/RqLop2AAAGoW3btmnEiBF655131NTUpNraWvl8PpWWlvYqTktLiz777DNZrVYtXbpUf/31lwzDUFJSUsjYnTt36vz588rLy1N6eroaGxv1+eefh4z7448/5HK5JEkFBQWyWq1yu93atWuXHj58qPnz52vUqFFyOp0yDENz5szR+PHjJUnjxo3r/YcBAEAfa2lpkWmaev/994P29/T0aOjQoZIe/9598803ys/PV05Ojh4+fCjDMPTVV19p06ZNslgsYc1F0Q4AwCBks9m0YcMGSVJhYaESEhJ04sQJLVy4UAkJCWHHOXDggPx+vz799FMlJydLkrKzs7V+/fqgcR6PR+fPn9e8efP07rvvSnpckFdUVOj69etBY/fv3y/TNPXFF19o2LBhkqS5c+fq66+/VlVVld566y0lJSUpKytLhmEoIyNDeXl5//SjAACgz3V1dSkqKkpbtmwJrFd/4pVXXpEkHTt2TImJiVqxYkXg2Nq1a7V69Wo1NTUpIyMjrLlY0w4AwCBUUFAQtP32229Lkurr68OOYZqmGhoaNH369EDBLkmjR4/WlClTgsa63e5nzltYWBi07ff7VVdXp2nTpsnv96ujoyPwyszMVGdnpzweT9g5AgDQH1JTU2Wapu7du6eUlJSg15Or0bq7u0O66U8KfL/fH/ZcdNoBABiEXn/99aDt1157TRaLRV6vN+wYHR0d6u7uDoklSSNHjgw6AdDa2iqLxRLy3NmUlJSQmD6fT6dOndKpU6eeOy8AAP2tq6tLd+7cCWx7vV41Nzdr6NChGjlypGbMmKEdO3Zo5cqVGjt2rDo6OnTp0iXZ7XZNnTpVU6dO1dGjR1VdXR24PH7fvn0aMWKExo4dG3YeFO0AALwEnj7T/7w1dKZpvvA8nnQWcnNzlZ+f/8wxdrv9hecBAMD/cvXq1cBNUSWpsrJSkpSfn681a9aopKREBw8eVGVlpdra2mS1WpWenq5p06ZJkiZOnKh169bpyJEjOnz4sOLj45WRkaGNGzcqLi4u7Dwo2gEAGIRu374d1PW+c+eO/H6/bDabhgwZIkny+XxB77l7927QttVqVVxcXODRNU+7detW0HZycrL8fr+8Xm9QZ/7pDsWTmAkJCTJNU5MnT/7bvyHcG/QAAPAiOBwOGYbx3OMxMTFyOp1BT0f5bzk5OcrJyfm/8mBNOwAAg9Dx48eDtmtrayVJmZmZSkxM1LBhw3TlypW/fU9UVJSmTJmiCxcuqLW1NbD/5s2bamhoCBqbmZn5zBjHjh0LiZmdna26ujq1tLSE5P30pfHx8fGSQk8uAADwMqHTDgDAIOT1erVlyxZlZmbq999/188//6wZM2YoNTVVkjR79mwdOnRIu3btUlpamq5cufLMjrrT6ZTb7dZHH32kuXPnyjRN1dbWasyYMUF3hU9LS1N2drZqamr04MGDwCPfnsR8umu+fPlyXb58WS6XS7Nnz9bo0aP14MEDeTweXbp0Sd9//72kx+vwhwwZopMnTyohIUHx8fFKT08PWTcPAMBgRqcdAIBBqKysTLGxsfrpp5908eJFFRYWqri4OHB8yZIlmjVrln755Rft3btXpmlq48aNIXHsdrtcLpesVqsMw9Dp06fldDo1ffr0kLGlpaUqKCjQxYsXtXfvXvX09KisrEySFBsbGxiXlJSkzZs3a+bMmaqrq9OePXtUU1Mjn8+noqKiwLiYmBitWbNGUVFR+u6777R9+3Y1Njb24acEAEDks/h7c695AACAXmhubtaGDRu0du1a5ebm9nc6AAAMOHTaAQBAn+ju7g7Zd/ToUVksFk2YMKEfMgIAYOBjTTsAAOgThw8flsfjkcPhUHR0tNxut+rr6zVnzhwlJyf3d3oAAAxIXB4PAAD6xK+//qqqqirdvHlTXV1dSk5OVl5enhYvXqzo6Oj+Tg8AgAGJoh0AAAAAgAjFmnYAAAAAACIURTsAAAAAABGKoh0AAAAAgAhF0Q4AAAAAQISiaAcAAAAAIEJRtAMAAAAAEKEo2gEAAAAAiFAU7QAAAAAARCiKdgAAAAAAItR/AJDPrhBHLEo8AAAAAElFTkSuQmCC"/>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Find-correlation-for-only-numeric-types">Find correlation for only numeric types<a class="anchor-link" href="#Find-correlation-for-only-numeric-types">¶</a></h3><h3 id="Methods-of-correlation-are-:">Methods of correlation are :<a class="anchor-link" href="#Methods-of-correlation-are-:">¶</a></h3><ul>
<li>pearson</li>
<li>kendall</li>
<li>spearman</li>
</ul>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Find correlation for only numeric types</span>
<span class="c1"># Methods of correlation are :</span>
<span class="c1"># - pearson</span>
<span class="c1"># - kendall</span>
<span class="c1"># - spearman</span>


<span class="nb">print</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">corr</span><span class="p">(</span><span class="n">numeric_only</span><span class="o">=</span><span class="kc">True</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">'We found that budget and gross has high correlation'</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>             year     score     votes    budget     gross   runtime
year     1.000000  0.097995  0.222945  0.291690  0.259504  0.120811
score    0.097995  1.000000  0.409182  0.061979  0.185583  0.399451
votes    0.222945  0.409182  1.000000  0.460932  0.632103  0.309212
budget   0.291690  0.061979  0.460932  1.000000  0.745881  0.273363
gross    0.259504  0.185583  0.632103  0.745881  1.000000  0.244360
runtime  0.120811  0.399451  0.309212  0.273363  0.244360  1.000000
We found that budget and gross has high correlation
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">correlation_matrix</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">corr</span><span class="p">(</span><span class="n">method</span><span class="o">=</span><span class="s1">'pearson'</span><span class="p">,</span> <span class="n">numeric_only</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">correlation_matrix</span><span class="p">,</span> <span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">"Correlation Matrix for Numeric Features"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">'Movie Fields'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">'Movie Fields'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,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JBKJjqJRjQkmERERERGRpuSZOju0RCLJt4TS2toaycnJSmXJyckwNTXVuPcS4DWYREREREREX7yyZcvizp07SmXBwcEoV65crtphgklERERERPQfk5aWhoiICERERADIWoYkIiIC8fHxAIBt27Zh+fLlivotW7ZEbGwstmzZgufPn+P48eO4fPky2rVrl6vjcogsERERERGRpgS5riPQyOPHjzFr1izF/U2bNgEAmjRpglGjRiExMVGRbAKAo6MjJk+ejI0bN+LIkSOws7PD8OHDc7VECQCIBEEQtHIGeoyzyH5ZOIvsl4WzyH5ZOIvsl4WzyH5ZOIvsl6VQzyIb80Bnx5YULa+zY2uKPZhERERERESakheOHkxd4TWYREREREREpBXswSQiIiIiItKQUEiuwdQV9mASERERERGRVjDBJCIiIiIiIq3gEFkiIiIiIiJNcZIftdiDSURERERERFrBHkwiIiIiIiJNcZIftdiDSURERERERFrBBJOIiIiIiIi0gkNkiYiIiIiINCXP1HUEeo09mERERERERKQV7MEkIiIiIiLSFCf5UYs9mERERERERKQV7MEkIiIiIiLSlJw9mOqwB5OIiIiIiIi0ggkmERERERERaQWHyBIREREREWlI4CQ/arEHk4iIiIiIiLSCPZhERERERESa4iQ/arEHk4iIiIiIiLSCCSYRERERERFpBYfIEhERERERaYqT/KjFHkwiIiIiIiLSCvZgEhERERERaUqeqesI9Bp7MImIiIiIiEgr2INJRERERESkKV6DqRZ7MImIiIiIiEgrmGASERERERGRVnCILBERERERkabkHCKrjl71YAqCgPj4eGRkZOg6FCIiIiIiIsolvUswx4wZg1evXuk6FCIiIiIiouwEue5uhYBeJZgGBgYoVqwYXr9+retQiIiIiIiIKJf0KsEEgN69e2PLli14+vSprkMhIiIiIiKiXNC7SX5WrFiB9PR0TJgwAWKxGEZGRkrb/f39dRQZERERERF98TjJj1p6l2AOGDBA1yEQERERERFRHuhdgunl5aXrEIiIiIiIiFQShExdh6DX9C7B/FhGRgZkMplSmZmZmY6iISIiIiIiInX0LsFMS0vD1q1bcfnyZZWzye7cuVMHUREREREREaHQLBeiK3o3i+yWLVtw9+5dDB48GBKJBMOHD0f37t1ha2uL0aNH6zo8vXfj9h2MmuiHph36oHLDNjj9d6CuQyINDB7aF8H3ziMmPgSnz+5FjZpV1Nbv1LkNrt88gZj4EARePYIWLb2Utjs42uGPVfNx/2EgXsTexd79/nAr46pUx9HRHqvX/o6wx1cQHXMHf1/8Cx06ttLymZGmOg3ogO2XN+P4o8P4I2ApKlQrn2Pddr3bYMnehTh4dx8O3t2H37fPU6pvKDbE0KmD8eepNTgSdhC7b+zAlMUTYVfUriBOhTTQrF9r/H5xJdY+2I4ZB+bCrap7jnWb9GyOqbt+wh9BG/FH0EZM3OKXrb6lvRUG/z4ai6+uxZrQbRi/8UcUdS2W36dBGmrTvy1WX1qHnWF7Me+v31G2atkc67bo1RJz9vyKzXe2Y/Od7Zi57SeV9Z3dnTHlzx+x5e4ObL+/G/MDFsK+uEN+ngZpaODgXrgWfBJPXt7C4VM7UK2GR451y1Vwx7pNi3Et+CReJIVgyIh+2eqM+X4Ijp7ZiYfPruPOwwvw37oMZdxd8/EMiD6P3iWY//zzDwYPHox69erB0NAQX331Fbp27YpevXrh4sWLug5P7717l4by7m6YNn6krkMhDXXp2g6/zJ2KeXOXorFnB9y9ex/7D2yAvYPqZKBO3Rr4038xNm/cjUYN2+PwoZPYtmMlvqpYTlFn2/ZVcC1dEr17DEOjhu3x7Nlz/BWwCWZmpoo6q9f+jrJl3dCz+1A0qNsWBw+ewIZNy1ClSsV8P2dS1rR9E4yYMQwbF23B0DYj8DgkHPO3zIW1nbXK+tXqV8WZv87i++4TMKrjd4iNjsNvW3+FvVPWa8bE1BhlK7tj8+ItGNZ6JGYMnYWSZZwxZ/3sAjwrykkdnwbo9eNA/LVkF/zaTcCzkEj8sGk6LOwsVdavUK8Srhy8iF97+eGnLlOR8CIeP2yeAZuitoo6362ZBMeSRbFkyK+Y0e4HvHoeh4lb/GBkalxQp0U5aNjeE19PH4ydi7djfLuxiAh9ghlbZsPKzkpl/Ur1PHDhr78xvcdUTO40AfHR8fDbMhu2Hz3fTi5O+GXvPDx/HIXpPabi+1ZjsHvpDkjTMwrqtCgHHTq3xsw5k7Bg3h9o1cQXIXfvY/u+NbCzt1VZ39TUBJERUZgzcyFiXsaprFO/YS34r9uOdi16oUfnwRCLxdixfx1MP/qfTqRPRIIgCLoO4mP9+vXDokWLYG9vj+HDh+OHH36Au7s7YmNjMX78eGzevDnXbUrjw/MhUv1XuWEbLJk7Hc0aN9B1KAXK3rWFrkPIldNn9+LmzWBMGD8LACASiRDy4CLWrNqERQtXZ6vvv3EpzMxM0aPbEEXZqTN7cOdOKL7/bjrKuLvi5u3TqFu7Ne6HPlS0+TD8KmbP/B2bNu4CADx/GYxxY2dg544DinaeRN6A34z5ijqFQQ1rN12H8Nn+CFiK+0FhWPrjcgBZz9fO69uw3/8Atq/49GUBBgYGOHhvH5b+uBwn9p5SWad81XJYdXgFetTpjdho1V9iCoNSYtVJWGEy48BcPAl6jM1+6wBkPd+LLq/GyY1HcXjl/k/uLzIwwMqgjdjstw6X9p1H0dLFMP/sckxtMRbPHz5TtLn0+p/Y89tWnN95Ol/PJz+lyAt/wjTvr9/xKOgh1s7I+jwXiURYe9UfRzYcwr4/9nxyfwMDA2y+sx1rZ6zCub1nAQDjlk9ApiwTS8YuzNfYC9qVlMe6DuGzHT61A7dv3sG0iXMAZD3f/9w7g/VrtmL54nVq970WfBJrV27C2pXqv+va2dng7uNL6Ny2H64E/qO12Avai6QQXYeQZ2k3D+rs2CY1Oujs2JrSux7MokWLIjY2FgBQokQJBAZmDfG8ceMGzM3NdRkakdZJJBJUq14Z585+GMosCALOnQ1E7TrVVe5Tu051nDt7Sans9OkLivrGxllrx6anpSu1mZ6egXr1aynKrl29iS5d28HGxgoikQhdfX1gbGKMixeuau386NPEEjHKeZTDPxduKsoEQcDNCzdRqYZmvcnGpsYQS8RIScp+3fp75hbmkMvleJPy9rNjprwzlIjhWrkM7l0KVpQJgoB7l4LhXqOcmj0/MDY1gqHEEG+S3gAAJEYSAFDqvRIEAdIMKcrW/kqL0VNuiSVilPFwR9DFIEWZIAgIvngb5WvkPAz+Y0amxkrPt0gkQi3vWogOf44Zm2dhw83NmPfX76jTsl6+nANpTiKRoEq1irhw/oqiTBAEXDh/GTXrVNPacSwsLQAAiYnJWmuTSJv0LsH08vJCREQEAKBjx444fvw4+vTpg40bN6JDh09n7FKpFKmpqUo3In1lZ2cDsViM2Nh4pfK42HgULar6WpqiRe0RG/cqx/phD8Lx9Olz+M36AdbWlpBIJBj7/VA4OxeDk9OHNgf2HwOJRIyIZzcRlxCKRUt+Rt9eIxAeHqnlsyR1rGytYCg2RGJcolJ5YnwibB1tNGpj2NTBiH/5Cv9cvKlyu8RYgmFTB+PMX2eR+oafibpkYWMBQ7EhkuOTlMqT45Jh5WCtURvdJ/dDUkwiQv6fpL54/BzxUXHoNrEvzCzNYSgRo+3wTrArbg9rDV9DlD8sbC3//3wrv7+T4pNg7aDZc9N/ykAkxiQg6OJtAICVvRVMi5ihy0hf3Dp3EzP7zsDV41cwac0UVKpbWdunQLlga2cNsViMuGz/01/B0dFeK8cQiUSYPXcyrl3+Bw9CH2mlTcoDQa67WyGgd7PI+vj4KP6uUqUKFi9ejPDwcDg5OcHFxeWT++/fvx979igPOdn6x69aj5NIX8lkMvTrPRLL/piLyKhbkMlkOHc2ECeOn4NIJFLUmzZ9HKysLNHBpx9exSegXfsW8N+0DG1a9UDIvTAdngHlRq9RPdC0oxe+7/YDpOnSbNsNxYbwWzkdEImwaMpSHURI2tRuRGfUbd8Qv/b0UzzfmbJMLBs+H4Pmj8TK4E3IlGXi3qVgBJ29iY/e8lQIdRnpC88OjTC9+1TF8y0yyOobuHbiKgL+/AsAEBHyBOVrVkCrvq1x7+pdncVL+W/u79NRoWJZdGzdV9ehEOVI7xLMj2VkZMDBwQEODprPita5c2elJBUAkPpSy5ERacerV4mQyWTZftl0cLRHTIzq6+RiYuLh+K8JgP5d//btu2jUoD0sLYtAYmSEV/EJOH12L27dugMAKF26FIYN7690nebdu/fRoEFtDBnaD99/N12bp0lqJCckI1OWCZt/9WbY2NsgITYxh72ydB/mi94je2J8r0kID32Sbbuh2BB+q36Ek7MjxnWfwN5LPfA68TUyZZmwsrdWKrdysEJyXJLafdsM6YB2Izpjfp9ZeHZfeaRBxN1wzGj7A0wtzCCWiPE6ISXrWs/gwn9NW2H2OiHl/8+38vvb2t4aSXHq398dh3ZGlxFd4ddnOiLvRyi1KZPK8OzhU6X6UY+e4avanKRNlxJeJUEmk8Eh2/90u2wjlfJizvxpaN6qCTq3648X0TGf3R5RftG7IbJyuRx79uzBsGHD0L9/f8TEZL2BduzYgTNnznxyf4lEAjMzM6Ubkb6SSqW4fesumnh9mIhJJBKhiVd9XL92S+U+16/dUqoPAE2beqqsn5LyBq/iE+BWxhXVa3jgyKGsCWBMzUwAZL3fPpaZmQkDA737WPhPk0llCLsThhqeH665FYlEqOFZHfdu5jwBQs8R3dHvu76Y2G8qwoKz9zi/Ty6dXUtgfM9Jaq/PpIKTKZUh4u5jVGzwYdkCkUiEig2q4NHNnEcOtB3WER3G+GLBgJ8QcSfnpPHd61S8TkhBUddiKO1RBrdOXtdq/JQ7MqkMj+88QpWGH5aeEolE8GhYFQ9uPshxv07Du6Dbtz0wu/9MPA5WHgYpk8rwKOghSpRxViovXroE4qIK7wRe/wVSqRTBt0Pg2eTD9bAikQiejevhn2u3P6vtOfOnoY1Pc3TrMAjPIp9/ZqT02eSZursVAnr3TXLfvn04f/48+vbtC7H4QwdrqVKlcPp04Z0Jr6Ckpr7D/bDHuB+W9QXkeXQM7oc9xouXsTqOjHKyYvl6DBjYA716d0G58mWwaMlPMDczw5YtWUO9V635HX4zf1DUX/nHBjRv0Rijx3yDsuXcMHnqt6heozLWrP4w61ynzm3g2aguXF1Lom275jhwcCMOHzqJM2eylvoJexCOx48isHjpz6hRswpKly6F0WO+QVNvTxwKOFmwDwBh95q98OnVFq18W6CUeyl8P/dbmJia4NjO4wCAKYsnYvDkQYr6PUf2wNc/DMBvP/yOl89ewsbBBjYONjD5/w8HhmJDzFo9A+WrlMOcMb/CwNBAUUcs0euBK1+EY+sC0KRXczTs6oViZUpgwJyhMDYzxoXdWT+iDl0wBt0m9lHUbzu8E7qM64U/J/6B+Kg4WDlYw8rBGsb/f74BoHbb+qhQrxIcShZF9Ra1MWHLDPxz4jruXgjKdnwqWAfXHUCLXq3Q1Ncbzu7OGPbLSJiYmeD0rqwf/L5d9D36TuqvqN95RFf0Ht8XyycsRWxUDKwdrGHtYK14fwPAgdX70NDHEy16tYSTSzG0GdAOtZvXwbHNRwr8/EjZ6hUb0Ke/L7r16oiy5dwwb6EfzMxNsWNr1gzRS1fNxdQZ3yvqSyQSVPKogEoeFSCRSOBUrCgqeVSAa+lSijpzf5+Orj3aY9SQCXjz5i0cHO3h4GgPExMuQ0T6Se++aZw/fx5Dhw6Fh4cH1q5dqyh3cXFBdHS0DiMrHO7ef4hBYyYp7s9ftgYA0LFNc8z5cbyuwiI19u09DDt7W0z9cSyKFrXHneBQdOn8NeJisybycS5ZTKmn8drVmxg86Hv8OH0cZswcj8ePI9G75wiEhnzo/Sjq5Ig5c6fB0dEOL1/GYcf2/Zj/63LFdplMBt+u32DW7AnYuXstzM3NEB4eieFDJ+DkiXMFdu6U5WzAeVjZWWPgDwNg62CDxyGPManfVCT+fyIYxxKOkMs/rCjVsZ8PjIyNMGuNn1I7GxZuwsaFm2HvZI+GrbJ6udedVF7qZmy38Qi6HAzSnWuHAmFpa4Uu3/eElYM1noY+we8DfkZKfNaMkLYl7CH/aAUx776tIDGWYMyqCUrt7F+8EwcWZy0pZO1og14/DoSVvRWSYpNwad85/LXs00tgUP67FHARlrZW6DmuD2wcbPAkJByz+/kpJnpyKO4A4aP3d+u+bSAxlmDS6ilK7exYtA07F20HAFw9fgWrp/6BLqO64ZtZQxH9+DnmD5uL0OuFd9mH/4qD+4/Bzt4WE6eOgYOjPe7duY/eXYch/v+T85VwVv6fXrSYA05d2Ke4P/LbQRj57SAEXryGrj4DAQADB/cCAOw7vEnpWN+NnIpd2w7k7wmRaoVksh1d0bt1MPv06YPFixfDwcEB/fv3x2+//YaiRYsiKioKU6ZM4TqY9EmFbR1M+jz/hXUwSXP/hXUwSXP/hXUwSXP/hXUwSXOFeh3Ma7t1dmyTOt10dmxN6V0PprOzM0JDQ7NN7HPlyhWULl1aR1EREREREREBkLMHUx29SzB9fX2xYsUKJCQkQBAEXL16FdHR0fj7778xefJkXYdHREREREREOdC7SX5q166NSZMm4c6dOzA2NsauXbvw/PlzTJo0CVWqVPl0A0RERERERKQTeteDuXz5cnh7e2P6dK7DR0REREREeoaT/KildwlmamoqfvrpJzg4OMDLywteXl6wtbXVdVhERERERET0CXqXYE6cOBEpKSn4+++/cf78eezevRseHh5o2rQpateurbQ2JhERERERUYHiJD9q6WW2ZmlpCR8fH/j4+CA8PBznzp3D8uXLYWJigkaNGqFVq1YoVqyYrsMkIiIiIiKij+jdJD8fS0xMRHBwMIKDg2FgYIDq1avj2bNnGDduHA4dOqTr8IiIiIiIiOgjeteDKZPJcOPGDZw7dw5BQUFwcXFB27Zt4enpCTMzMwDAtWvXsHLlSvj4+Og4WiIiIiIi+qJwiKxaepdgDhs2DHK5HA0bNsTcuXPh6uqarU6lSpUUySYRERERERHpB71LMAcMGIB69erByMgoxzrm5uZYsWJFAUZFREREREQECEKmrkPQa3qXYDZu3FjXIRAREREREVEe6PUkP0RERERERFR46F0PJhERERERkd7iJD9qsQeTiIiIiIiItII9mERERERERJoS2IOpDnswiYiIiIiISCvYg0lERERERKQpXoOpFnswiYiIiIiISCuYYBIREREREZFWcIgsERERERGRpjjJj1rswSQiIiIiIiKtYA8mERERERGRpjjJj1rswSQiIiIiIiKtYIJJREREREREWsEhskRERERERJriJD9qsQeTiIiIiIiItII9mERERERERJriJD9qsQeTiIiIiIiItII9mERERERERJpiD6Za7MEkIiIiIiIirWCCSURERERERFrBIbJERERERESa4jIlarEHk4iIiIiIiLSCPZhERERERESa4iQ/arEHk4iIiIiIiLSCCSYRERERERFpBYfIEhERERERaYqT/KjFHkwiIiIiIiLSCvZgEhERERERaYqT/KjFHkwiIiIiIiLSCvZgEhERERERaYrXYKrFHkwiIiIiIiLSCiaYREREREREpBUcIktERERERKQpTvKj1heRYNq7ttB1CFSA4iNO6joEKkCrq8/QdQhUgMZGn9V1CFSAqtiV1nUIVIAmW9bQdQhEpAVfRIJJRERERESkFezBVIvXYBIREREREZFWMMEkIiIiIiIireAQWSIiIiIiIk0Jgq4j0GvswSQiIiIiIiKtYA8mERERERGRpjjJj1rswSQiIiIiIiKtYA8mERERERGRptiDqRZ7MImIiIiIiEgrmGASERERERGRVnCILBERERERkaYEDpFVhz2YREREREREpBXswSQiIiIiItIUJ/lRiz2YREREREREpBVMMImIiIiIiEgrOESWiIiIiIhIU4Kg6wj0GnswiYiIiIiISCvYg0lERERERKQpTvKjFnswiYiIiIiISCvYg0lERERERKQp9mCqxR5MIiIiIiIi0gommERERERERKQVHCJLRERERESkKYFDZNVhDyYRERERERFpBXswiYiIiIiINCTIBV2HoLFjx44hICAASUlJcHFxwaBBg+Du7p5j/cOHD+PEiROIj4+HpaUl6tati969e8PIyEjjY7IHk4iIiIiI6D8mMDAQmzZtgq+v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OjqMjIiIiIiIvnSCnJP8qKN3CSYAWFtbw9raWqnM3d1dN8EQERERERGRRvQywSQiIiIiItJLheRaSF3Ru2swiYiIiIiIqHBigklERERERERawSGyREREREREmuIQWbXYg0lERERERERawR5MIiIiIiIiTQlcpkQd9mASERERERGRVjDBJCIiIiIiIq3gEFkiIiIiIiJNcZIftdiDSURERERERFrBHkwiIiIiIiINCezBVIs9mERERERERKQV7MEkIiIiIiLSFHsw1WIPJhEREREREWkFE0wiIiIiIiLSCg6RJSIiIiIi0pRcrusI9Bp7MImIiIiIiEgr2INJRERERESkKU7yoxZ7MImIiIiIiEgrmGASERERERGRVnCILBERERERkaY4RFYt9mASERERERGRVrAHk4iIiIiISEOCwB5MddiDSURERERERFrBHkwiIiIiIiJN8RpMtdiDSURERERERFrBBJOIiIiIiIi0gkNkiYiIiIiINMUhsmqxB5OIiIiIiIi0gj2YREREREREGhLYg6nWF5Fg1rB203UIVIBWV5+h6xCoAA27NVvXIVABcvKYrusQqAA1rhil6xCoAE1/nK7rEIhICzhEloiIiIiIiLTii+jBJCIiIiIi0goOkVWLPZhERERERESkFezBJCIiIiIi0pRc1wHoN/ZgEhERERERkVawB5OIiIiIiEhDXKZEPfZgEhERERERkVYwwSQiIiIiIiKt4BBZIiIiIiIiTXGIrFrswSQiIiIiIiKtYA8mERERERGRprhMiVrswSQiIiIiIiKtYIJJREREREREWsEhskRERERERBriOpjqsQeTiIiIiIiItII9mERERERERJriJD9qsQeTiIiIiIiItIIJJhEREREREWkFh8gSERERERFpiJP8qMceTCIiIiIiItIK9mASERERERFpipP8qMUeTCIiIiIiItIK9mASERERERFpSGAPplrswSQiIiIiIiKtYIJJREREREREWsEhskRERERERJriEFm12INJREREREREWsEeTCIiIiIiIg1xkh/12INJREREREREWsEEk4iIiIiIiLSCQ2SJiIiIiIg0xSGyarEHk4iIiIiIiLSCPZhEREREREQa4iQ/6rEHk4iIiIiIiLRC73sw5XI5nj59Cnt7exQpUkTX4RARERER0ReMPZjq6V2CuWHDBpQqVQre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igRz1KhR+RoUERERERGRPvpolUZSQeN1MPv06YOLFy8q7mdkZGDPnj2IjY3Nl8CIiIiIiIiocNE4wfy39PR07N69mwkmERERERF9MeSCSGe3wiDPCSYRERERERHRx5hgEhERERERkVZwHUwiIiIiIiINcR1M9XKVYAYEBODSpUsAgMzMTADAjh07YGFhka2uSCTCxIkTtRAiERERERERFQYaJ5j29vZ48+YN3rx5o1SWmJiIxMTEbPW5TiYREREREf3XcJkS9TROMFesWJGfcRAREREREVEhx2swiYiIiIiINFRYlgvRFc4iS0RERERERFrBBJOIiIiIiIi04osYIltKbKnrEKgAjY0+q+sQqAA5eUzXdQhUgDre+UnXIVABylg6VdchUAFKepSm6xCINMJlStRjDyYRERERERFpxRfRg0lERERERKQNnORHvc9KMKVSKZ48eYLk5GSUL18elpYcikpERERERPSlynOCeeTIEezevRupqakAgOnTp6Ny5cpISUnB999/jz59+sDb21trgRIREREREZF+y9M1mGfPnsXGjRtRrVo1jBgxQmmbpaUlKlWqhMDAQK0ESEREREREpC8EHd4KgzwlmIcOHUKtWrXw3XffoWbNmtm2u7m54dmzZ58dHBERERERERUeeRoi+/LlS7Rp0ybH7UWKFMGbN2/yHBQREREREZE+4iQ/6uWpB9PMzAwpKSk5bo+KioK1tXVeYyIiIiIiIqJCKE8JZvXq1XH69Gm8ffs227Znz57h9OnTKofOEhERERERFWaCINLZrTDI0xDZnj17Ytq0aRg/frwikTx37hzOnDmDq1evwsbGBr6+vloNlIiIiIiIiPRbnhJMW1tb/Prrr9i+fbtittgLFy7AxMQEDRs2RJ8+fbgmJhERERER0Rcmz+tgWllZYfjw4Rg+fDhSUlIgl8thaWkJA4M8jbolIiIiIiLSe3JdB6Dn8pxgfoy9lURERERERKRRgrlnzx4AQJcuXWBgYKC4/ym8DpOIiIiIiP5LBBSOyXZ0RaMEc/fu3QCATp06wcDAQHH/U5hgEhERERERfTk0SjB37typ9j4RERERERGRVq7BJCIiIiIi+hLIBV1HoN/yNOXrwoULce3aNUilUm3HQ0RERERERIVUnnowHzx4gKtXr8LExAS1atVCgwYNULVqVYjF7BAlIiIiIqL/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DbLvxr7vS2lRVJIk7UXr3X5/NO64qitc3Uuf97x6vabnPuec7+m4y/d+n/M8qrGCSURERERERGrBCiYREREREVEJcZIf1VjBJCIiIiIiIrVgBZOIiIiIiKiEZAJNR6DdWMEkIiIiIiIitWCCSURERERERGrBIbJEREREREQlJONCJSqxgklERERERERqwQomERERERFRCbF+qRormERERERERKQWTDCJiIiIiIhILThEloiIiIiIqIRkmg5Ay7GCSURERERERGrBCiYREREREVEJcZkS1VjBJCIiIiIiIrVgBZOIiIiIiKiEWL9UjRVMIiIiIiIiUgsmmERERERERKQWHCJLRERERERUQlymRDVWMImIiIiIiEgtWMEkIiIiIiIqIS5TohormERERERERKQWTDCJiIiIiIhILThEloiIiIiIqIQ4QFY1VjCJiIiIiIhILVjBJCIiIiIiKiEuU6IaK5hERERERESkFqxgEhERERERlZCcd2GqxAomERERERERqQUTTCIiIiIiIlILDpElIiIiIiIqIU7yo5rWVTDz8/ORl5en+D0pKQmHDx9GUFCQBqMiIiIiIiKiN9G6CubixYvh7u6ODh06ICsrC9OmTYOuri4yMzMxaNAgdOjQQdMhEhERERFRGSXjJD8qaV0F8+HDh6hduzYA4OrVqzA3N8fy5csxZswYHD16VMPRERERERERUXG0LsHMy8uDoaEhACAoKAju7u4QCoWoWbMmkpKSNBwdERERERERFUfrEkxbW1tcv34dycnJCAoKQv369QEAmZmZisSTiIiIiIhIE+Qa/PkYaF2C6evri82bN2P06NGoWbMmnJycABRUM6tVq6bh6IiIiIiIiKg4WjfJT7NmzeDs7Iy0tDRUqVJF0e7i4gJ3d3cNRkZERERERGUdJ/lRTesSTAAwNzeHubk5ACA7OxshISGws7NDpUqVNBsYERERERERFUvrEsxff/0VderUgbe3N/Lz8zF16lQkJiYCAL755hs0a9ZMwxESERERERFRUbTuHsx79+7B2dkZAHD9+nXI5XJs2LABQ4YMwZ49ezQcHRERERERlWUyDf58DLSugpmdnQ1jY2MAQGBgIJo2bQqRSISGDRti8+bNGo5OszoN7IzPR/SEubUFYu49xNpZq/Ag6EGRfT/7sgM8e7WDQ62C+1ijgiOxddGmQv3t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dAhERERERESFcB1M1bQywQwLC1P5OBNQIiIiIiIi7aOVCebcuXNVPs57MImIiIiISBNkH8lkO5qilQnm+vXrlX6XSCSIiYnBjh070KdPHw1FRURERERERKpoZYJpZGRUqM3V1RW6urrYuHEjFi1apIGoiIiIiIiorOM9mKpp5TqYxTEzM0N8fLymwyAiIiIiIqIiaGUF89GjR0q/y+VypKenY9++fahatapmgiIiIiIiIiKVtDLBnDRpUpHtNWvWxKhRo0o5GiIiIiIiogIyDpFVSesSTIlEgjp16mD48OHQ09MDAAgEApiamkJfX1/D0REREREREVFxtC7B1NXVRWxsLIRCIaytrTUdDhERERERkQIn+VFNKyf5adWqFU6dOqXpMIiIiIiIiOgtaF0FEwBkMhmOHz+O4OBgODo6QiQSKT0+aNAgDUVGRERERERExdHKBPPx48dwdHQEADx9+lTD0RARERERERWQgUNkVdHKBHP27NmaDoGIiIiIiIjeklYmmERERERERNqIk/yoppWT/BAREREREdHHhxVMIiIiIiKiEpKxgqkSK5hERERERESkFkwwiYiIiIiISC04RJaIiIiIiKiE5FymRCVWMImIiIiIiEgtWMEkIiIiIiIqIU7yoxormERERERERKQWTDCJiIiIiIhILThEloiIiIiIqITkHCKrEiuYREREREREpBasYBIREREREZUQlylRjRVMIiIiIiIiUgsmmERERERERKQWHCJLRERERERUQpzkRzVWMImIiIiIiEgtWMEkIiIiIiIqIVYwVWMFk4iIiIiIiNSCFUwiIiIiIqISYv1SNVYwiYiIiIiISC2YYBIREREREZFaCOS8S/WTJBaLsXfvXvTo0QN6enqaDoc+MF7vsoXXu2zh9S5beL3LFl5v+hSxgvmJEovF2LVrF8RisaZDoVLA61228HqXLbzeZQuvd9nC602fIiaYREREREREpBZMMImIiIiIiEgtmGASERERERGRWjDB/ETp6enB19eXN4yXEbzeZQuvd9nC61228HqXLbze9CniLLJERERERESkFqxgEhERERERkVowwSQiIiIiIiK1YIJJREREREREasEEk4iISA3mzJmDDRs2qHWfiYmJ8PPzQ0xMjFr3S0Tax8/PD9evX9d0GETvjQkmEdFHhh9C6H2NHj0ahw8f1nQYRGVS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w9jYGC4uLm88Rt++fREaGorp06fDy8sL9vb2ePHiBaKjoxEcHIz169e/6ykWqU6dOmjfvj327duHR48ewdXVFTo6OkhISMCVK1cwZMiQYu9J9fLywrFjx7Bs2TJER0fDwsIC58+fL7QcSkhICNatW4dmzZrBzs4OUqkU58+fVyTUREREnyImmEREpJJQKMTkyZOxY8cOXL58GWfOnIGNjQ369++Prl27Fupfvnx5ODk5ISIiAu3atSt2PclXmZubY+HChdi1axeuXbuGf/75ByYmJqhcuTL69ev3IU4LX3/9NRwdHXHy5Els27YNOjo6sLa2RqtWrVCrVq1itxOJRJg1axbWrVuHY8eOQV9fH61atYKbmxsWLlyo6Fe1alXUr18ft27dwokTJyASiVClShVMmzYNTk5OH+SciIiINE0gfzm7AhEREREREdF74D2YREREREREpBZMMImIiIiIiEgtmGASERERERGRWjDBJCIiIiIiIrVggklERERERERqwQSTiIiIiIiI1IIJJhEREREREakFE0wiIiIiIiJSCyaYREREREREpBZMMImIiIiIiEgtmGASERERERGRWjDBJCIiIiIiIrX4P/vIkR87bW2uAAAAAElFTkSuQmCC"/>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Convert-Object-type-feature-like-Company-Features-to-numeric-or-categoric-type">Convert Object type feature like Company Features to numeric or categoric type<a class="anchor-link" href="#Convert-Object-type-feature-like-Company-Features-to-numeric-or-categoric-type">¶</a></h3>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Convert Object type feature like Company Features to numeric or categoric type</span>

<span class="n">df_normalized</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>

<span class="k">for</span> <span class="n">col</span> <span class="ow">in</span> <span class="n">df_normalized</span><span class="o">.</span><span class="n">columns</span><span class="p">:</span>
    <span class="k">if</span> <span class="n">df_normalized</span><span class="p">[</span><span class="n">col</span><span class="p">]</span><span class="o">.</span><span class="n">dtype</span> <span class="o">==</span> <span class="s1">'object'</span><span class="p">:</span>
        <span class="n">df_normalized</span><span class="p">[</span><span class="n">col</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_normalized</span><span class="p">[</span><span class="n">col</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s1">'category'</span><span class="p">)</span>
        <span class="n">df_normalized</span><span class="p">[</span><span class="n">col</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_normalized</span><span class="p">[</span><span class="n">col</span><span class="p">]</span><span class="o">.</span><span class="n">cat</span><span class="o">.</span><span class="n">codes</span>

<span class="n">df_normalized</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[ ]:</div>
<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html" tabindex="0">
<div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>name</th>
<th>rating</th>
<th>genre</th>
<th>year</th>
<th>released</th>
<th>score</th>
<th>votes</th>
<th>director</th>
<th>writer</th>
<th>star</th>
<th>country</th>
<th>budget</th>
<th>gross</th>
<th>company</th>
<th>runtime</th>
<th>yearcorrect</th>
</tr>
</thead>
<tbody>
<tr>
<th>5445</th>
<td>533</td>
<td>5</td>
<td>0</td>
<td>2009</td>
<td>696</td>
<td>7.8</td>
<td>1100000.0</td>
<td>1155</td>
<td>1778</td>
<td>2334</td>
<td>55</td>
<td>237000000</td>
<td>2847246203</td>
<td>2253</td>
<td>162.0</td>
<td>29</td>
</tr>
<tr>
<th>7445</th>
<td>535</td>
<td>5</td>
<td>0</td>
<td>2019</td>
<td>183</td>
<td>8.4</td>
<td>903000.0</td>
<td>162</td>
<td>743</td>
<td>2241</td>
<td>55</td>
<td>356000000</td>
<td>2797501328</td>
<td>1606</td>
<td>181.0</td>
<td>39</td>
</tr>
<tr>
<th>3045</th>
<td>6896</td>
<td>5</td>
<td>6</td>
<td>1997</td>
<td>704</td>
<td>7.8</td>
<td>1100000.0</td>
<td>1155</td>
<td>1778</td>
<td>1595</td>
<td>55</td>
<td>200000000</td>
<td>2201647264</td>
<td>2253</td>
<td>194.0</td>
<td>17</td>
</tr>
<tr>
<th>6663</th>
<td>5144</td>
<td>5</td>
<td>0</td>
<td>2015</td>
<td>698</td>
<td>7.8</td>
<td>876000.0</td>
<td>1125</td>
<td>2550</td>
<td>524</td>
<td>55</td>
<td>245000000</td>
<td>2069521700</td>
<td>1540</td>
<td>138.0</td>
<td>35</td>
</tr>
<tr>
<th>7244</th>
<td>536</td>
<td>5</td>
<td>0</td>
<td>2018</td>
<td>192</td>
<td>8.4</td>
<td>897000.0</td>
<td>162</td>
<td>743</td>
<td>2241</td>
<td>55</td>
<td>321000000</td>
<td>2048359754</td>
<td>1606</td>
<td>149.0</td>
<td>38</td>
</tr>
<tr>
<th>...</th>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
<tr>
<th>3818</th>
<td>3360</td>
<td>6</td>
<td>4</td>
<td>2000</td>
<td>1119</td>
<td>6.5</td>
<td>5200.0</td>
<td>730</td>
<td>1123</td>
<td>2319</td>
<td>54</td>
<td>20500000</td>
<td>1400</td>
<td>477</td>
<td>103.0</td>
<td>21</td>
</tr>
<tr>
<th>7625</th>
<td>6720</td>
<td>-1</td>
<td>4</td>
<td>2019</td>
<td>1149</td>
<td>5.7</td>
<td>320.0</td>
<td>2546</td>
<td>2565</td>
<td>1915</td>
<td>55</td>
<td>20500000</td>
<td>790</td>
<td>2308</td>
<td>104.0</td>
<td>39</td>
</tr>
<tr>
<th>7580</th>
<td>4664</td>
<td>3</td>
<td>5</td>
<td>2019</td>
<td>1835</td>
<td>5.2</td>
<td>735.0</td>
<td>1445</td>
<td>2203</td>
<td>2278</td>
<td>55</td>
<td>20500000</td>
<td>682</td>
<td>1992</td>
<td>93.0</td>
<td>40</td>
</tr>
<tr>
<th>2417</th>
<td>3406</td>
<td>-1</td>
<td>6</td>
<td>1993</td>
<td>85</td>
<td>7.3</td>
<td>5100.0</td>
<td>33</td>
<td>1718</td>
<td>2563</td>
<td>27</td>
<td>11900000</td>
<td>596</td>
<td>796</td>
<td>134.0</td>
<td>13</td>
</tr>
<tr>
<th>3203</th>
<td>6990</td>
<td>5</td>
<td>4</td>
<td>1997</td>
<td>2811</td>
<td>5.7</td>
<td>5800.0</td>
<td>961</td>
<td>229</td>
<td>2758</td>
<td>55</td>
<td>15000000</td>
<td>309</td>
<td>821</td>
<td>85.0</td>
<td>17</td>
</tr>
</tbody>
</table>
<p>7668 rows × 16 columns</p>
</div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">df_normalized</span><span class="o">.</span><span class="n">sort_values</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="p">[</span><span class="s1">'gross'</span><span class="p">],</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">ascending</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="n">df_normalized</span><span class="o">.</span><span class="n">corr</span><span class="p">())</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">'We found that budget and gross has high correlation'</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>                 name    rating     genre      year  released     score  \
name         1.000000 -0.008069  0.016355  0.011453 -0.011311  0.017097   
rating      -0.008069  1.000000  0.072423  0.008779  0.016613 -0.001314   
genre        0.016355  0.072423  1.000000 -0.081261  0.029822  0.027965   
year         0.011453  0.008779 -0.081261  1.000000 -0.000695  0.097995   
released    -0.011311  0.016613  0.029822 -0.000695  1.000000  0.042788   
score        0.017097 -0.001314  0.027965  0.097995  0.042788  1.000000   
votes        0.013088  0.033225 -0.145307  0.222945  0.016097  0.409182   
director     0.009079  0.019483 -0.015258 -0.020795 -0.001478  0.009559   
writer       0.009081 -0.005921  0.006567 -0.008656 -0.002404  0.019416   
star         0.006472  0.013405 -0.005477 -0.027242  0.015777 -0.001609   
country     -0.010737  0.081244 -0.037615 -0.070938 -0.020427 -0.133348   
budget       0.020921 -0.108776 -0.328484  0.291690  0.011120  0.061979   
gross        0.006601 -0.097213 -0.233385  0.259504  0.000806  0.185583   
company      0.009211 -0.032943 -0.071067 -0.010431 -0.010474  0.001030   
runtime      0.010392  0.062145 -0.052711  0.120811  0.000868  0.399451   
yearcorrect  0.010225  0.006403 -0.078210  0.996397 -0.003775  0.106295   

                votes  director    writer      star   country    budget  \
name         0.013088  0.009079  0.009081  0.006472 -0.010737  0.020921   
rating       0.033225  0.019483 -0.005921  0.013405  0.081244 -0.108776   
genre       -0.145307 -0.015258  0.006567 -0.005477 -0.037615 -0.328484   
year         0.222945 -0.020795 -0.008656 -0.027242 -0.070938  0.291690   
released     0.016097 -0.001478 -0.002404  0.015777 -0.020427  0.011120   
score        0.409182  0.009559  0.019416 -0.001609 -0.133348  0.061979   
votes        1.000000  0.000260  0.000892 -0.019282  0.073625  0.460932   
director     0.000260  1.000000  0.299067  0.039234  0.017490 -0.003584   
writer       0.000892  0.299067  1.000000  0.027245  0.015343 -0.030641   
star        -0.019282  0.039234  0.027245  1.000000 -0.012998 -0.018534   
country      0.073625  0.017490  0.015343 -0.012998  1.000000  0.082334   
budget       0.460932 -0.003584 -0.030641 -0.018534  0.082334  1.000000   
gross        0.632103 -0.014758 -0.023064 -0.001529  0.093994  0.745881   
company      0.133204  0.004404  0.005646  0.012442  0.095548  0.167250   
runtime      0.309212  0.017624 -0.003511  0.010174 -0.078412  0.273363   
yearcorrect  0.218289 -0.020385 -0.008391 -0.027606 -0.079009  0.284099   

                gross   company   runtime  yearcorrect  
name         0.006601  0.009211  0.010392     0.010225  
rating      -0.097213 -0.032943  0.062145     0.006403  
genre       -0.233385 -0.071067 -0.052711    -0.078210  
year         0.259504 -0.010431  0.120811     0.996397  
released     0.000806 -0.010474  0.000868    -0.003775  
score        0.185583  0.001030  0.399451     0.106295  
votes        0.632103  0.133204  0.309212     0.218289  
director    -0.014758  0.004404  0.017624    -0.020385  
writer      -0.023064  0.005646 -0.003511    -0.008391  
star        -0.001529  0.012442  0.010174    -0.027606  
country      0.093994  0.095548 -0.078412    -0.079009  
budget       0.745881  0.167250  0.273363     0.284099  
gross        1.000000  0.155786  0.244360     0.252749  
company      0.155786  1.000000  0.034402    -0.014144  
runtime      0.244360  0.034402  1.000000     0.120636  
yearcorrect  0.252749 -0.014144  0.120636     1.000000  
We found that budget and gross has high correlation
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">correlation_matrix</span> <span class="o">=</span> <span class="n">df_normalized</span><span class="o">.</span><span class="n">corr</span><span class="p">(</span><span class="n">method</span><span class="o">=</span><span class="s1">'pearson'</span><span class="p">,</span> <span class="n">numeric_only</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">correlation_matrix</span><span class="p">,</span> <span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">"Correlation Matrix for Numeric Features"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">'Movie Fields'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">'Movie Fields'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,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"/>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Lets-find-the-correlation-between-features-which-have-above-0.5-value">Lets find the correlation between features which have above 0.5 value<a class="anchor-link" href="#Lets-find-the-correlation-between-features-which-have-above-0.5-value">¶</a></h3>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Lets find the correlation between features which have above 0.5 value </span>


<span class="n">sorted_corr</span> <span class="o">=</span> <span class="n">df_normalized</span><span class="o">.</span><span class="n">corr</span><span class="p">()</span><span class="o">.</span><span class="n">unstack</span><span class="p">()</span><span class="o">.</span><span class="n">sort_values</span><span class="p">()</span>
<span class="n">high_corr</span> <span class="o">=</span> <span class="n">sorted_corr</span><span class="p">[(</span><span class="n">sorted_corr</span><span class="p">)</span> <span class="o">&gt;</span> <span class="mf">0.5</span><span class="p">]</span>
<span class="n">high_corr</span> <span class="o">=</span> <span class="n">high_corr</span><span class="p">[(</span><span class="n">high_corr</span><span class="p">)</span> <span class="o">&lt;</span> <span class="mf">0.9</span><span class="p">]</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">high_corr</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[ ]:</div>
<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain" tabindex="0">
<pre>gross   votes     0.632103
votes   gross     0.632103
gross   budget    0.745881
budget  gross     0.745881
dtype: float64</pre>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h4 id="We-found-that-there-is-high-correlation-between-'Gross'-feature-and-'Votes'-feature-as-well-as-with-'Gross'-feature-and-'Budget'-features">We found that there is high correlation between 'Gross' feature and 'Votes' feature as well as with 'Gross' feature and 'Budget' features<a class="anchor-link" href="#We-found-that-there-is-high-correlation-between-'Gross'-feature-and-'Votes'-feature-as-well-as-with-'Gross'-feature-and-'Budget'-features">¶</a></h4>
</div>
</div>
</div>
</div>
</main>
</body>
</html>