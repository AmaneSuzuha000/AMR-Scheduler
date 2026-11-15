$log = "E:\Remote\项目\AMR\tools\word_export.log"
"[$(Get-Date -Format o)] start" | Out-File $log -Encoding utf8
try {
  $src = "E:\Remote\项目\AMR\docs\论文\论文-主文档.docx"
  $word = New-Object -ComObject Word.Application
  $word.Visible = $false
  $word.DisplayAlerts = 0
  $doc = $word.Documents.Open($src, $false, $false)
  "opened, toc=" + $doc.TablesOfContents.Count | Out-File $log -Append -Encoding utf8
  if ($doc.TablesOfContents.Count -gt 0) { $doc.TablesOfContents.Item(1).Update() }
  $doc.Fields.Update() | Out-Null
  $doc.Repaginate()
  "pages=" + $doc.ComputeStatistics(2) | Out-File $log -Append -Encoding utf8
  $doc.Save()
  $doc.Close(0)
  $word.Quit()
  "done" | Out-File $log -Append -Encoding utf8
} catch {
  "ERR: " + $_.Exception.Message | Out-File $log -Append -Encoding utf8
  try { $word.Quit() } catch {}
}