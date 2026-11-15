$log = "E:\Remote\项目\AMR\tools\pdf_export.log"
"start" | Out-File $log -Encoding utf8
try {
  $src = "E:\Remote\项目\AMR\docs\论文\论文-主文档.docx"
  $pdf = "E:\Remote\项目\AMR\docs\论文\论文-主文档.pdf"
  $word = New-Object -ComObject Word.Application
  $word.Visible = $false
  $word.DisplayAlerts = 0
  $doc = $word.Documents.Open($src, $false, $true)
  "opened" | Out-File $log -Append -Encoding utf8
  $doc.SaveAs2($pdf, 17)
  "saved pdf" | Out-File $log -Append -Encoding utf8
  $doc.Close(0)
  $word.Quit()
  "done" | Out-File $log -Append -Encoding utf8
} catch {
  "ERR: " + $_.Exception.Message | Out-File $log -Append -Encoding utf8
  try { $word.Quit() } catch {}
}